"""
Core Model: MTS-O1-Embedder
Combines TS Encoder, Projector, and LLM Backbone.
"""
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers import AutoModelForCausalLM
from .ts_encoder import TimeSeriesEncoder, PatchTokenizer, ChannelMixer
from .projector import TimeSeriesProjector
from .patch_encoder import TCFormer


class _TSOnlyStub(nn.Module):
    """Lightweight nn.Module placeholder for ts_only_mode=True.

    Carries no trainable parameters — it purely provides a `.config.hidden_size`
    attribute so that MTSEmbedder can size the projector/norms to `output_dim`
    without loading 4 B LLM weights.  All trainer code that iterates
    `llm.parameters()` or calls `llm.to(device)` works correctly on an empty module.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)


class LatentAttentionPooling(nn.Module):
    """NV-Embed style Latent Attention Pooling (Li et al., ICLR 2025 Spotlight).

    K learnable latent queries cross-attend to all token hidden states and produce
    a single fixed-size embedding by averaging over the K query outputs.
    Empirically outperforms mean/last-token pooling on retrieval tasks.

    Args:
        d_model: hidden dimension (= LLM hidden_size)
        num_latents: number of learnable query vectors (K)
        n_heads: attention heads in the cross-attention layer
    """
    def __init__(self, d_model: int, num_latents: int = 8, n_heads: int = 8) -> None:
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, d_model]
            attention_mask: [B, S] with 1 for valid, 0 for padding (optional)
        Returns:
            [B, d_model] global representation
        """
        B = hidden_states.shape[0]
        target_dtype = self.cross_attn.in_proj_weight.dtype
        hidden_states = hidden_states.to(target_dtype)
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1).to(target_dtype)  # [B, K, d]
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True = ignore
        pooled, _ = self.cross_attn(
            query=queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        pooled = self.norm(pooled)  # [B, K, d]
        return pooled.mean(dim=1)   # [B, d]


class ForecastingHead(nn.Module):
    """Linear head that maps a fixed-size embedding to a multi-step forecast.

    Input:  [B, input_dim]
    Output: [B, ts_input_dim, forecast_horizon]  (channel-first, normalized space)
    """
    def __init__(self, input_dim: int, ts_input_dim: int, forecast_horizon: int) -> None:
        super().__init__()
        self.ts_input_dim = ts_input_dim
        self.forecast_horizon = forecast_horizon
        self.linear = nn.Linear(input_dim, ts_input_dim * forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).reshape(x.shape[0], self.ts_input_dim, self.forecast_horizon)


class MTSEmbedder(nn.Module):
    def get_text_embeds(self, text_input_ids: torch.Tensor) -> torch.Tensor:
        # Handle simple model or PEFT wrapped model
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
            return self.llm.model.embed_tokens(text_input_ids)
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model.model, "model") and hasattr(self.llm.base_model.model.model, "embed_tokens"):
            return self.llm.base_model.model.model.embed_tokens(text_input_ids)

        if hasattr(self.llm, "get_input_embeddings"):
            embedding_layer = self.llm.get_input_embeddings()
            if embedding_layer is not None:
                return embedding_layer(text_input_ids)

        raise AttributeError("Could not find embed_tokens layer in the LLM.")

    @staticmethod
    def _patch_gdr_if_needed(model: nn.Module) -> None:
        """Patch FLA modules that use Triton kernels incompatible with this environment.

        Root cause: system Triton 3.1.0 has OOB memory access in FLA's Triton kernels
        for chunk_gated_delta_rule.  The "Falling back to torch implementation" message
        at startup is about causal-conv1d (optional conv op), NOT about
        chunk_gated_delta_rule — that Triton kernel still runs and writes out-of-bounds,
        causing cudaErrorIllegalAddress surfacing asynchronously in downstream layers.

        Proper fix: install FLA's custom triton-nightly (see requirements.txt Step 2).
        Code fix: replace chunk_gated_delta_rule on every GDR module with FLA's naive
        PyTorch GPU implementation, bypassing Triton entirely.

        We also replace FusedRMSNormGated: that Triton norm kernel also crashes.
        """
        # ── PyTorch RMSNorm+SiLU gate — replaces fla's FusedRMSNormGated ─────
        class _RMSNormGated(nn.Module):
            def __init__(self, weight, bias, eps):
                super().__init__()
                self.weight = weight
                self.bias = bias
                self.eps = eps

            def forward(self, x, z=None):
                orig_dtype = x.dtype
                x_f = x.float()
                rstd = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
                out = (x_f * rstd).to(orig_dtype)
                if self.weight is not None:
                    out = out * self.weight.to(dtype=orig_dtype)
                if self.bias is not None:
                    out = out + self.bias.to(dtype=orig_dtype)
                if z is not None:
                    out = out * torch.nn.functional.silu(z.to(dtype=orig_dtype))
                return out

        # ── Pure-PyTorch GDR recurrence (bf16 native + TBPTT-1, no Triton) ─────
        # Fallback ONLY if FLA naive_chunk is unavailable.
        # Qwen3.5 passes [B,T,H,D] tensors WITHOUT a head_first parameter, so we
        # always expect [B,T,H,D] input and convert to head-first internally.
        def _gdr_pytorch_gpu(q, k, v, beta, g=None, scale=-1,
                             initial_state=None, output_final_state=False,
                             chunk_size=64, head_first=True,
                             use_qk_l2norm_in_kernel=False, **kwargs):
            import torch.nn.functional as _F
            orig_dtype = q.dtype
            # Qwen3.5 always calls with [B,T,H,D] (head-last). Convert to [B,H,T,D].
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            if beta is not None and beta.dim() == 3: beta = beta.transpose(1, 2).contiguous()
            if g    is not None and g.dim()    == 3: g    = g.transpose(1, 2).contiguous()
            B, H, T, DK = q.shape; DV = v.shape[-1]; dtype = q.dtype
            if use_qk_l2norm_in_kernel:
                q = _F.normalize(q.float(), dim=-1).to(dtype)
                k = _F.normalize(k.float(), dim=-1).to(dtype)
            if float(scale) <= 0: scale = DK ** -0.5
            q = (q.float() * float(scale)).to(dtype)
            q_seq = q.permute(2,0,1,3).contiguous()
            k_seq = k.permute(2,0,1,3).contiguous()
            v_seq = v.permute(2,0,1,3).contiguous()
            beta_seq = (None if beta is None else
                        (beta.permute(2,0,1).contiguous() if beta.dim()==3
                         else beta.permute(2,0,1,3).contiguous()))
            g_seq = (None if g is None else
                     (g.permute(2,0,1).contiguous() if g.dim()==3
                      else g.permute(2,0,1,3).contiguous()))
            S = (initial_state.to(dtype).contiguous() if initial_state is not None
                 else torch.zeros(B, H, DK, DV, dtype=dtype, device=q_seq.device))
            outs = []
            for t in range(T):
                S = S.detach()
                q_t, k_t, v_t = q_seq[t], k_seq[t], v_seq[t]
                if g_seq is not None:
                    decay = g_seq[t].float().exp().to(dtype)
                    S = S * (decay.unsqueeze(-1).unsqueeze(-1) if decay.dim()==2
                             else decay.unsqueeze(-1))
                o_t = torch.einsum('bhd,bhdv->bhv', q_t, S)
                outs.append(o_t)
                kS = torch.einsum('bhd,bhdv->bhv', k_t, S)
                delta = v_t - kS
                if beta_seq is not None:
                    b_t = beta_seq[t]
                    if b_t.dim() == 2:
                        delta = delta * b_t.unsqueeze(-1)
                        outer = torch.einsum('bhd,bhv->bhdv', k_t, delta)
                    else:
                        outer = torch.einsum('bhd,bhv->bhdv', k_t, delta) * b_t.unsqueeze(-1)
                else:
                    outer = torch.einsum('bhd,bhv->bhdv', k_t, delta)
                S = S + outer
            # Output is [B,H,T,V] (head-first); convert back to [B,T,H,V] for Qwen3.5
            o = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
            return o, S if output_final_state else None

        # ── Patch FLA module-level namespace (critical) ────────────────────────
        # FLA's GatedDeltaNet.forward() calls chunk_gated_delta_rule via the
        # module-level name imported at the top of fla/layers/delta_net.py.
        # Setting module.chunk_gated_delta_rule (instance attr) is NOT enough —
        # we must replace the name in every FLA sys.module that holds it.
        norm_replaced = 0
        gdr_replaced = 0

        _naive_gdr = None
        try:
            from fla.ops.gated_delta_rule.naive import naive_gated_delta_rule as _naive_gdr_import
            _naive_gdr = _naive_gdr_import
        except ImportError:
            pass

        # If fla.ops.gated_delta_rule.naive is not available (server environment),
        # fall back to the pure-PyTorch GPU implementation defined above so that
        # the Triton kernel is never called (it crashes with OOB at ~sample 164).
        _gdr_patch_fn = _naive_gdr if _naive_gdr is not None else _gdr_pytorch_gpu

        import sys as _sys
        for _mod_name, _mod in list(_sys.modules.items()):
            if _mod is None:
                continue
            if 'fla' not in _mod_name:
                continue
            # Patch chunk_gated_delta_rule at module level
            if hasattr(_mod, 'chunk_gated_delta_rule'):
                setattr(_mod, 'chunk_gated_delta_rule', _gdr_patch_fn)
                gdr_replaced += 1

        # Also set on each GDR module instance (belt-and-suspenders)
        for module in model.modules():
            if hasattr(module, 'chunk_gated_delta_rule'):
                module.chunk_gated_delta_rule = _gdr_patch_fn

            # Replace fla's FusedRMSNormGated Triton norm with PyTorch.
            if hasattr(module, 'norm') and 'fla' in type(module.norm).__module__:
                orig = module.norm
                module.norm = _RMSNormGated(
                    weight=getattr(orig, 'weight', None),
                    bias=getattr(orig, 'bias', None),
                    eps=getattr(orig, 'eps', 1e-6),
                )
                norm_replaced += 1

        if norm_replaced:
            print(f" -> [norm patch] Replaced fla FusedRMSNormGated with PyTorch "
                  f"in {norm_replaced} layer(s).")
        if gdr_replaced:
            src = "naive" if _naive_gdr is not None else "pytorch_gpu fallback"
            print(f" -> [gdr patch] Patched chunk_gated_delta_rule in {gdr_replaced} FLA "
                  f"module namespace(s) with {src} (avoids Triton OOB).")
        else:
            print(" -> [gdr patch] No chunk_gated_delta_rule found in FLA modules; skipping.")

    def _load_llm(
        self,
        llm_model_path: str,
        allow_llm_fallback: bool,
        llm_attn_implementation: str,
        load_to_cpu: bool = False,
    ) -> nn.Module:
        print(f"Loading LLM from {llm_model_path}...")
        try:
            print(f" -> Checking local cache for {llm_model_path}...")
            # Load directly onto the target GPU so weights stream from disk into VRAM
            # without staging through CPU RAM.  device_map={"": local_rank} keeps the
            # entire model on one device (safe for PEFT LoRA training, unlike "auto").
            # In torchrun DDP, LOCAL_RANK ensures each process uses its own GPU.
            _local_rank = int(os.environ.get("LOCAL_RANK", 0))
            _world_size = int(os.environ.get("WORLD_SIZE", 1))
            _is_ddp = _world_size > 1

            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "attn_implementation": llm_attn_implementation,
                "dtype": torch.bfloat16,
            }
            # Under DDP, skip device_map — accelerate's dispatch hooks conflict with
            # DDP's device management, causing CUDA illegal memory access when tensors
            # end up under different device-placement regimes.  DDP wrapper handles
            # device placement instead.  Single-GPU keeps device_map for fast loading.
            # When load_to_cpu=True (ts_only_embedding mode), keep LLM on CPU to save GPU VRAM.
            if _is_ddp:
                print(f" -> DDP mode (world_size={_world_size}): loading to CPU, DDP will move to cuda:{_local_rank}.")
            elif load_to_cpu:
                print(f" -> ts_only mode: loading LLM to CPU to preserve GPU VRAM for ts_encoder/projector.")
            else:
                load_kwargs["device_map"] = {"": _local_rank}
                print(f" -> Loading directly to cuda:{_local_rank} (device_map, skips CPU staging).")
            print(f" -> Attention implementation: {llm_attn_implementation}")
            print(f" -> Loading model weights in bfloat16 (halves VRAM vs fp32 default).")
            try:
                model = AutoModelForCausalLM.from_pretrained(llm_model_path, **load_kwargs)
            except TypeError:
                # Backward compatibility: older transformers may not support this kwarg.
                print(" -> 'attn_implementation' not supported by this transformers version, retrying without it.")
                load_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(llm_model_path, **load_kwargs)
            # Under DDP, model stays on CPU here.  DDP wrapper (trainer.py) will
            # move the full MTSEmbedder (LLM + ts_encoder + projector) to the
            # correct device via device_ids, ensuring a single consistent placement.
            print(f" -> Successfully loaded {llm_model_path} from local cache.")
            # Detect and fix model.language_model.* key prefix mismatch.
            # Qwen3.5 checkpoints saved with the multimodal wrapper use this prefix,
            # while the current transformers text-only architecture expects model.* directly.
            import glob as _glob
            _shard_files = sorted(_glob.glob(os.path.join(llm_model_path, "*.safetensors")))
            if _shard_files:
                try:
                    from safetensors.torch import load_file as _load_sf
                    _probe_sd = _load_sf(_shard_files[0], device="cpu")
                    _probe_key = next(iter(_probe_sd.keys()))
                    del _probe_sd
                    if "language_model" in _probe_key:
                        print("[weight remap] Checkpoint uses model.language_model.* — remapping to model.* ...")
                        _sd: dict = {}
                        for _sf in _shard_files:
                            for _k, _v in _load_sf(_sf, device="cpu").items():
                                _sd[_k.replace("model.language_model.", "model.")] = _v.to(torch.bfloat16)
                        _miss, _unex = model.load_state_dict(_sd, strict=False)
                        del _sd
                        print(f"[weight remap] Done. Missing={len(_miss)}, Unexpected={len(_unex)}")
                        # Tie lm_head ← embed_tokens (Qwen convention for tied weights)
                        if _miss and hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
                            model.lm_head.weight = model.model.embed_tokens.weight
                            print("[weight remap] Tied lm_head.weight → model.embed_tokens.weight")
                except Exception as _e:
                    print(f"[weight remap] Skipped: {_e}")
            self._patch_gdr_if_needed(model)
            # Patch loss_function directly on the model instance.
            # transformers stores `self.loss_function = ForCausalLMLoss` at __init__ time,
            # so patching the module-level name in main.py has no effect after the model is
            # already constructed.  Replacing the instance attribute avoids the logits.float()
            # cast that allocates ~7 GiB for a [B, T, 151936] logit tensor.
            import torch.nn.functional as _F_lm
            def _bf16_lm_loss(logits, labels, vocab_size, **kwargs):
                shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
                shift_labels = labels[..., 1:].contiguous().view(-1)
                return _F_lm.cross_entropy(
                    shift_logits, shift_labels.to(shift_logits.device), ignore_index=-100
                )
            if hasattr(model, "loss_function"):
                model.loss_function = _bf16_lm_loss
                print("[lm loss patch] Instance loss_function patched — logits stay in bf16.")
            return model
        except Exception as exc:
            if not allow_llm_fallback:
                raise RuntimeError(
                    f"Failed to load local LLM from '{llm_model_path}'. "
                    "Formal runs require a valid local model. "
                    "Use allow_llm_fallback=True only for debugging."
                ) from exc

            print(f" -> Not found in local cache: {exc}")
            print(" -> Debug fallback enabled. Using a tiny random LLM.")
            from transformers import LlamaConfig, LlamaForCausalLM

            fallback_config = LlamaConfig(
                vocab_size=32000,
                hidden_size=512,
                num_hidden_layers=2,
                num_attention_heads=4
            )
            return LlamaForCausalLM(fallback_config)

    def _encode_ts_tokens(self, ts_input: torch.Tensor) -> torch.Tensor:
        ts_feat = self.ts_encoder(ts_input)
        if self.channel_mixer is not None:
            # Infer num_channels from ts_encoder's channel-first normalization.
            # ts_encoder always normalizes to [B, C, T]; for 3-D inputs, C is dim-1
            # when shape[1] <= shape[2] (channel-first heuristic) else dim-2.
            raw = ts_input
            if raw.ndim == 2:
                raw = raw.unsqueeze(1)
            if raw.ndim == 3:
                c1, c2 = raw.shape[1], raw.shape[2]
                num_channels = c1 if c1 <= c2 else c2
            else:
                num_channels = raw.shape[1]
            ts_feat = self.channel_mixer(ts_feat, num_channels)
        ts_embeds = self.projector(ts_feat)
        return self.projector_norm(ts_embeds)

    def _prepare_multimodal_inputs(
        self,
        ts_input: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        text_input_ids = text_input_ids.long()
        if attention_mask is None:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.long, device=text_input_ids.device)

        ts_embeds = self._encode_ts_tokens(ts_input)
        text_embeds = self.get_text_embeds(text_input_ids)
        inputs_embeds, full_attention_mask, full_labels = self.insert_ts_embeddings(
            text_embeds,
            text_input_ids,
            ts_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return text_input_ids, ts_embeds, inputs_embeds, full_attention_mask, full_labels

    @staticmethod
    def _make_bidirectional_4d_mask(
        attention_mask_1d: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert 1D padding mask [B, S] to additive 4D bidirectional mask [B, 1, S, S].

        Unlike the model's internal prepare_4d_causal_attention_mask, this version does
        NOT apply the lower-triangular causal constraint: every non-padding position can
        attend to every other non-padding position.  Safe only with standard MHA models
        (Qwen2.5 / LLaMA).  Do NOT use with Qwen3.5 GDR layers.
        """
        B, S = attention_mask_1d.shape
        # key_mask[b, 0, i, j] = 0.0 if position j is valid else -inf
        key_valid = attention_mask_1d.to(dtype=torch.float32).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        key_valid = key_valid.expand(B, 1, S, S)
        mask_4d = (1.0 - key_valid) * torch.finfo(dtype).min
        return mask_4d.to(dtype)

    def _project_pooled_embedding(
        self,
        last_hidden_state: torch.Tensor,
        text_input_ids: torch.Tensor,
        ts_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: Optional[str] = None,
    ) -> torch.Tensor:
        effective_pooling = pooling or self.embedding_pooling

        # Latent Attention Pooling: latent queries cross-attend to ALL hidden states
        if effective_pooling == "latent" and self.latent_pooler is not None:
            embedding = self.latent_pooler(last_hidden_state, attention_mask=attention_mask)
            embedding = self.final_norm(embedding.to(self.final_norm.weight.dtype))
            embedding = embedding.to(self.final_projection.weight.dtype)
            return self.final_projection(embedding)

        ts_mask = None
        if effective_pooling == "ts_tokens":
            ts_mask = self._get_ts_token_mask(
                text_input_ids,
                ts_embeds.shape[1],
                last_hidden_state.shape[1],
            )

        embedding = self._pool_hidden_state(
            last_hidden_state,
            attention_mask=attention_mask,
            pooling=pooling,
            ts_mask=ts_mask,
        )
        embedding = self.final_norm(embedding.to(self.final_norm.weight.dtype))  # P3: stabilise before projection
        embedding = embedding.to(self.final_projection.weight.dtype)
        return self.final_projection(embedding)

    def _get_inner_decoder(self):
        """Return the inner decoder model (e.g. Qwen3_5Model) stripped of CausalLM head.

        Bypasses ``output_hidden_states=True`` on the ForCausalLM wrapper, which has a
        known interaction with gradient checkpointing in some transformers builds that
        produces NaN hidden states on the first forward pass.

        Works for both PEFT-wrapped models (PeftModel → LoraModel → ForCausalLM → Decoder)
        and plain models (ForCausalLM → Decoder).
        """
        llm = self.llm
        # Unwrap PEFT: PeftModel → base_model (LoraModel) → model (ForCausalLM)
        if hasattr(llm, 'base_model') and hasattr(llm.base_model, 'model'):
            llm = llm.base_model.model
        # Unwrap ForCausalLM → inner Decoder (e.g. Qwen3_5Model)
        if hasattr(llm, 'model'):
            return llm.model
        return llm  # fallback: already the decoder

    def _compute_embedding(
        self,
        ts_input: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: Optional[str] = None,
        require_grad: bool = False,
    ) -> torch.Tensor:
        text_input_ids, ts_embeds, inputs_embeds, full_attention_mask, _ = self._prepare_multimodal_inputs(
            ts_input,
            text_input_ids,
            attention_mask=attention_mask,
        )

        # Call the inner decoder model directly to get last_hidden_state without
        # output_hidden_states=True, which causes NaN when combined with gradient
        # checkpointing in some transformers builds (GC recompute path is incorrect
        # for the all-hidden-states tuple in ForCausalLM).  The inner decoder has
        # LoRA adapters applied in-place and GC enabled, so behaviour is equivalent.
        decoder = self._get_inner_decoder()

        # Bidirectional attention (Phase 2 / NV-Embed): override the model's default
        # causal mask by passing a pre-computed 4D mask that only masks padding tokens,
        # letting all non-padding positions attend to each other in both directions.
        # SAFE ONLY with standard MHA models (Qwen2.5 / LLaMA).  Qwen3.5 GDR layers
        # use a recurrent kernel that requires causal ordering — keep 1D mask for those.
        attn_mask_for_decoder = full_attention_mask
        if self.use_bidirectional_attn and full_attention_mask is not None and full_attention_mask.dim() == 1:
            attn_mask_for_decoder = self._make_bidirectional_4d_mask(
                full_attention_mask.unsqueeze(0),
                dtype=inputs_embeds.dtype,
            ).squeeze(0)
        elif self.use_bidirectional_attn and full_attention_mask is not None and full_attention_mask.dim() == 2:
            attn_mask_for_decoder = self._make_bidirectional_4d_mask(
                full_attention_mask,
                dtype=inputs_embeds.dtype,
            )

        with torch.set_grad_enabled(require_grad):
            outputs = decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask_for_decoder,
                return_dict=True,
            )

        return self._project_pooled_embedding(
            outputs.last_hidden_state,
            text_input_ids,
            ts_embeds,
            attention_mask=full_attention_mask,
            pooling=pooling,
        )

    def __init__(
        self,
        llm_model_path: str,
        ts_hidden_dim: int = 128,
        encoder_base_channels: int = 64,
        encoder_dropout: float = 0.1,
        encoder_norm: str = "group",
        embedding_pooling: str = "mean",
        ts_marker_start_id: int = 151652,
        ts_marker_end_id: int = 151653,
        output_dim: int = 1024,
        llm_attn_implementation: str = "eager",
        stem_strides: Optional[Sequence[int]] = None,
        patch_size: int = 5,
        encoder_type: str = "cnn",
        patch_stride: Optional[int] = None,
        allow_llm_fallback: bool = False,
        projector_hidden_dim: Optional[int] = None,
        channel_mixer: bool = False,
        channel_mixer_heads: int = 4,
        # TC-Former specific parameters
        tc_former_layers: int = 4,
        tc_former_heads: int = 4,
        tc_former_ff_dim: int = 512,
        tc_former_max_channels: int = 64,
        tc_former_max_patches: int = 256,
        tc_former_use_revin: bool = True,
        # Phase 2: bidirectional attention (NV-Embed) — use only with MHA models (Qwen2.5)
        use_bidirectional_attn: bool = False,
        # Phase 2: latent attention pooling (NV-Embed) — activated when embedding_pooling="latent"
        latent_pooling_num_latents: int = 8,
        latent_pooling_heads: int = 8,
        # Forecasting head (U+G unified training)
        ts_input_dim: int = 1,       # number of input channels (needed for ForecastingHead output size)
        forecast_horizon: int = 0,   # 0 = disabled; > 0 = attach ForecastingHead
        forecast_channels: Optional[int] = None,  # raw output channels for ForecastingHead; defaults to ts_input_dim
        # When True, load the LLM onto CPU instead of GPU to preserve GPU VRAM
        # (useful when ts_only_embedding=True and the GPU has limited free memory)
        llm_load_to_cpu: bool = False,
        # When True, skip LLM loading entirely (ts_only training/eval only).
        # llm_dim is set to output_dim, making the projector ts_hidden_dim→output_dim.
        # Saves 30-60s startup time, ~8 GB RAM, and allows much smaller checkpoints.
        ts_only_mode: bool = False,
    ) -> None:
        super().__init__()
        resolved_stem_strides = list(stem_strides) if stem_strides is not None else [5, 5]
        self.allow_llm_fallback = allow_llm_fallback
        if ts_only_mode:
            # No LLM weights loaded; stub provides .config.hidden_size = output_dim
            self.llm = _TSOnlyStub(hidden_size=output_dim)
            print(f" -> ts_only_mode: LLM skipped entirely. llm_dim={output_dim} (=output_dim)")
        else:
            self.llm = self._load_llm(llm_model_path, allow_llm_fallback, llm_attn_implementation,
                                      load_to_cpu=llm_load_to_cpu)

        self.llm_dim = self.llm.config.hidden_size

        if encoder_type == "tc_former":
            self.ts_encoder = TCFormer(
                d_model=ts_hidden_dim,
                n_layers=tc_former_layers,
                n_heads=tc_former_heads,
                d_ff=tc_former_ff_dim,
                dropout=encoder_dropout,
                patch_size=patch_size,
                max_channels=tc_former_max_channels,
                max_patches=tc_former_max_patches,
                use_revin=tc_former_use_revin,
            )
            param_count = sum(p.numel() for p in self.ts_encoder.parameters()) / 1e6
            print(f" -> TS Encoder: TC-Former (d={ts_hidden_dim}, L={tc_former_layers}, "
                  f"h={tc_former_heads}, ff={tc_former_ff_dim}, patch={patch_size}, "
                  f"RevIN={tc_former_use_revin}, params={param_count:.1f}M)")
        elif encoder_type == "patch":
            self.ts_encoder = PatchTokenizer(
                hidden_dim=ts_hidden_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                dropout=encoder_dropout,
            )
            print(f" -> TS Encoder: PatchTokenizer (patch_size={patch_size}, patch_stride={patch_stride or patch_size})")
        else:
            self.ts_encoder = TimeSeriesEncoder(
                hidden_dim=ts_hidden_dim,
                base_channels=encoder_base_channels,
                stem_strides=resolved_stem_strides,
                patch_size=patch_size,
                dropout=encoder_dropout,
                norm_type=encoder_norm,
            )
            print(f" -> TS Encoder: CNN (stem_strides={resolved_stem_strides}, patch_size={patch_size})")

        self.projector = TimeSeriesProjector(ts_dim=ts_hidden_dim, llm_dim=self.llm_dim,
                                              hidden_dim=projector_hidden_dim)
        if projector_hidden_dim is not None:
            print(f" -> Projector: {ts_hidden_dim}→{projector_hidden_dim}→{self.llm_dim} (slim)")
        else:
            print(f" -> Projector: {ts_hidden_dim}→{self.llm_dim*2}→{self.llm_dim} (default expansion 2×)")

        self.channel_mixer: Optional[ChannelMixer] = None
        if channel_mixer and encoder_type != "tc_former":
            # TC-Former has built-in channel attention; ChannelMixer is redundant
            self.channel_mixer = ChannelMixer(ts_hidden_dim, num_heads=channel_mixer_heads,
                                              dropout=encoder_dropout)
            print(f" -> ChannelMixer: {channel_mixer_heads} heads on {ts_hidden_dim}-dim features")

        self.projector_norm = nn.LayerNorm(self.llm_dim)
        self.output_dim = output_dim
        self.final_norm = nn.LayerNorm(self.llm_dim)   # stabilise pooled LLM hidden state
        self.final_projection = nn.Linear(self.llm_dim, self.output_dim)

        self.embedding_pooling = embedding_pooling
        self.use_bidirectional_attn = use_bidirectional_attn
        self.ts_marker_start_id = ts_marker_start_id
        self.ts_marker_end_id = ts_marker_end_id

        # Phase 2: Latent Attention Pooling (activated when embedding_pooling="latent")
        self.latent_pooler: Optional[LatentAttentionPooling] = None
        if embedding_pooling == "latent":
            self.latent_pooler = LatentAttentionPooling(
                d_model=self.llm_dim,
                num_latents=latent_pooling_num_latents,
                n_heads=latent_pooling_heads,
            )
            print(f" -> Latent Attention Pooling: {latent_pooling_num_latents} queries, "
                  f"{latent_pooling_heads} heads")
        if use_bidirectional_attn:
            print(" -> Bidirectional attention: enabled (causal mask removed — use only with MHA models)")

        # Forecasting head: embedding → [B, C, H] future prediction
        self.ts_input_dim = ts_input_dim
        _forecast_c = forecast_channels if forecast_channels is not None else ts_input_dim
        self.forecasting_head: Optional[ForecastingHead] = None
        if forecast_horizon > 0:
            self.forecasting_head = ForecastingHead(output_dim, _forecast_c, forecast_horizon)
            print(f" -> ForecastingHead: {output_dim}→{_forecast_c}×{forecast_horizon} "
                  f"(C={_forecast_c}, H={forecast_horizon})")

    def _get_ts_token_mask(self, text_input_ids, ts_seq_len, max_seq_len):
        """Return float mask (B, max_seq_len) with 1.0 at TS token positions.

        TS tokens are inserted immediately after the ts_marker_start token.
        This mask is used by the 'ts_tokens' pooling mode to pool only the LLM hidden
        states corresponding to TS tokens, reducing text-identity bias.
        """
        B = text_input_ids.shape[0]
        mask = torch.zeros(B, max_seq_len, device=text_input_ids.device, dtype=torch.float32)
        for i in range(B):
            start_pos = (text_input_ids[i] == self.ts_marker_start_id).nonzero(as_tuple=True)[0]
            if len(start_pos) > 0:
                s = start_pos[0].item() + 1  # EEG tokens start right after the start marker
                mask[i, s : min(s + ts_seq_len, max_seq_len)] = 1.0
            else:
                # Fallback: EEG tokens were prepended (no markers in prompt)
                mask[i, :ts_seq_len] = 1.0
        return mask

    def _pool_hidden_state(self, hidden_state, attention_mask=None, pooling=None, ts_mask=None):
        pooling = pooling or self.embedding_pooling

        if pooling == "ts_tokens":
            if ts_mask is not None:
                mask = ts_mask.unsqueeze(-1).to(hidden_state.dtype)
                denom = mask.sum(dim=1).clamp_min(1.0)
                return (hidden_state * mask).sum(dim=1) / denom
            # Fallback to mean if ts_mask not provided
            pooling = "mean"

        if pooling == "last":
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
                batch_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)
                return hidden_state[batch_indices, lengths]
            return hidden_state[:, -1, :]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (hidden_state * mask).sum(dim=1) / denom

        return hidden_state.mean(dim=1)

    def insert_ts_embeddings(self, text_embeds, text_input_ids, ts_embeds, attention_mask=None, labels=None):
        """
        Insert TS embeddings between ts_marker_start and ts_marker_end tokens.
        Adjusts attention mask and labels if provided.
        """
        batch_size = text_embeds.shape[0]
        ts_seq_len = ts_embeds.shape[1]
        
        new_inputs_embeds = []
        new_attention_masks = []
        new_labels = []
        
        for i in range(batch_size):
            start_idx = (text_input_ids[i] == self.ts_marker_start_id).nonzero(as_tuple=True)[0]
            end_idx = (text_input_ids[i] == self.ts_marker_end_id).nonzero(as_tuple=True)[0]
            
            if len(start_idx) > 0 and len(end_idx) > 0 and start_idx[0] < end_idx[0]:
                s = start_idx[0].item()
                e = end_idx[0].item()
                
                # Slices for embeds
                prefix_e = text_embeds[i, :s+1]
                suffix_e = text_embeds[i, e:]
                new_embeds = torch.cat([prefix_e, ts_embeds[i], suffix_e], dim=0)
                new_inputs_embeds.append(new_embeds)
                
                # Slices for attention mask
                if attention_mask is not None:
                    prefix_a = attention_mask[i, :s+1]
                    suffix_a = attention_mask[i, e:]
                    ts_a = torch.ones((ts_seq_len,), dtype=attention_mask.dtype, device=attention_mask.device)
                    new_masks = torch.cat([prefix_a, ts_a, suffix_a], dim=0)
                    new_attention_masks.append(new_masks)
                
                # Slices for labels
                if labels is not None:
                    prefix_l = labels[i, :s+1]
                    suffix_l = labels[i, e:]
                    ts_l = torch.full((ts_seq_len,), -100, dtype=labels.dtype, device=labels.device)
                    new_label = torch.cat([prefix_l, ts_l, suffix_l], dim=0)
                    new_labels.append(new_label)
            else:
                # Fallback to concat if markers missing
                new_inputs_embeds.append(torch.cat([ts_embeds[i], text_embeds[i]], dim=0))
                if attention_mask is not None:
                    ts_a = torch.ones((ts_seq_len,), dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attention_masks.append(torch.cat([ts_a, attention_mask[i]], dim=0))
                if labels is not None:
                    ts_l = torch.full((ts_seq_len,), -100, dtype=labels.dtype, device=labels.device)
                    new_labels.append(torch.cat([ts_l, labels[i]], dim=0))
                    
        # Pad to max length
        max_len = max(t.shape[0] for t in new_inputs_embeds)
        
        padded_embeds = torch.zeros((batch_size, max_len, text_embeds.shape[-1]), device=text_embeds.device, dtype=text_embeds.dtype)
        padded_masks = torch.zeros((batch_size, max_len), device=text_embeds.device, dtype=torch.long) if attention_mask is not None else None
        padded_labels = torch.full((batch_size, max_len), -100, device=text_embeds.device, dtype=torch.long) if labels is not None else None
            
        for i in range(batch_size):
            l = new_inputs_embeds[i].shape[0]
            padded_embeds[i, :l] = new_inputs_embeds[i]
            if attention_mask is not None:
                padded_masks[i, :l] = new_attention_masks[i]
            if labels is not None:
                padded_labels[i, :l] = new_labels[i]
                
        return padded_embeds, padded_masks, padded_labels
        
    def freeze_llm(self):
        print("Freezing full LLM backbone...")
        for param in self.llm.parameters():
            param.requires_grad = False
            
    def apply_lora(self, r=8, lora_alpha=16, lora_dropout=0.05):
        """
        Apply LoRA (Low-Rank Adaptation) to the LLM backbone for memory-efficient training.
        """
        print("Applying LoRA to the LLM backbone...")
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            # Target projection matrices in the attention mechanism.
            # Covers both standard (Qwen2/3) and Qwen3.5 GDR hybrid layers:
            #   q/k/v/o_proj  — shared by full-attn and GDR linear-attn blocks
            #   gate_proj / up_proj / down_proj — MLP SwiGLU
            #   g_proj        — GDR gate projection (Qwen3.5 GatedDeltaNet)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj", "g_proj"]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
            # Full gradient checkpointing for ALL layers (including 24 GDR layers).
            # GDR layers use our CPU-loop _gdr_pytorch replacement, which is deterministic
            # and sync-safe — GC recomputation during backward is clean and correct.
            # This is simpler than selective/MLP-only GC and avoids async CUDA interactions.
            self.llm.enable_input_require_grads()
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(f" -> Gradient checkpointing enabled (all layers, use_reentrant=False).")
        except ImportError:
            print("Warning: 'peft' library not found. Falling back to smart_freeze_llm.")
            self.smart_freeze_llm()
            
    def smart_freeze_llm(self, num_trainable_layers=2):
        """
        Freeze all but the last `num_trainable_layers` of the LLM.
        This allows adaptation without destroying the base knowledge.
        """
        print(f"Smart Freezing: Keeping last {num_trainable_layers} layers trainable...")
        
        # 1. Freeze everything first
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze Head (if present and distinct, though typically tied)
        # For Qwen/Llama, 'lm_head'
        if hasattr(self.llm, "lm_head"):
             for param in self.llm.lm_head.parameters():
                 param.requires_grad = True
                 
        # 3. Unfreeze Norm
        if hasattr(self.llm.model, "norm"):
             for param in self.llm.model.norm.parameters():
                 param.requires_grad = True
                 
        # 4. Unfreeze last N decoder layers
        # structure: self.llm.model.layers (ModuleList)
        layers = self.llm.model.layers
        num_layers = len(layers)
        start_idx = num_layers - num_trainable_layers
        
        for i in range(start_idx, num_layers):
            print(f"  -> Unfreezing Layer {i}")
            for param in layers[i].parameters():
                param.requires_grad = True

    def unfreeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = True

    def forward(self, ts_input, text_input_ids, attention_mask=None, labels=None, return_embedding=False):
        """
        ts_input: [Batch, Dim, Time] or [Batch, Time] (raw time series)
        text_input_ids: [Batch, Seq_Len] (tokenized text)
        attention_mask: [Batch, Seq_Len]
        return_embedding: if True, return (None, embedding) instead of LM outputs
        """
        if return_embedding:
            # Use _compute_embedding (inner decoder path) to avoid NaN from
            # output_hidden_states=True with gradient checkpointing.
            embedding = self._compute_embedding(
                ts_input, text_input_ids, attention_mask=attention_mask, require_grad=True
            )
            return None, embedding

        text_input_ids, ts_embeds, inputs_embeds, full_attention_mask, full_labels = self._prepare_multimodal_inputs(
            ts_input,
            text_input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # No output_hidden_states=True: that flag combined with gradient checkpointing
        # produces NaN hidden states in some transformers builds (GC recompute path).
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
        )
        return outputs

    def get_gallery_embedding(self, full_ts: torch.Tensor) -> torch.Tensor:
        """Embed full trajectory (history+future) for asymmetric bi-encoder gallery.

        Gallery encoder sees [C, T+H] (history concatenated with future).
        At retrieval time, query encoder sees [C, T] (history only), so the
        similarity between query and gallery is future-informed: windows with
        similar histories AND similar futures land close together.

        Uses the same ts_encoder/projector weights as get_ts_only_embedding;
        variable input length is handled transparently by TCFormer / CNN encoder.
        """
        ts_norm = self._encode_ts_tokens(full_ts)
        emb = ts_norm.mean(dim=1)   # [B, llm_dim]
        return self.final_projection(emb)

    def get_ts_only_embedding(self, ts_input):
        """Embed TS using only the TS encoder and projector — no LLM forward.

        Eliminates the text-identity bias that occurs when all samples share
        nearly identical context strings. The retrieval embedding is derived
        purely from signal content: ts_encoder → projector → mean-pool over tokens.

        Gradient flow is determined by the caller's context (torch.no_grad() at
        inference, or leave gradients enabled during training).
        Returns [B, output_dim] embedding.
        """
        ts_norm = self._encode_ts_tokens(ts_input)
        emb = ts_norm.mean(dim=1)              # [B, llm_dim]
        return self.final_projection(emb)

    def get_embedding_for_training(self, ts_input, text_input_ids, attention_mask=None, pooling=None):
        """
        Get embeddings with gradients flowing — used during training for contrastive loss.
        Uses the same context-only (retrieval) prompt as get_embedding(), so the contrastive
        objective directly optimizes the retrieval embedding space.
        """
        return self._compute_embedding(
            ts_input,
            text_input_ids,
            attention_mask=attention_mask,
            pooling=pooling,
            require_grad=True,
        )

    def get_embedding(self, ts_input, text_input_ids, attention_mask=None, pooling=None):
        """
        Get the final dense embedding for retrieval.
        """
        return self._compute_embedding(
            ts_input,
            text_input_ids,
            attention_mask=attention_mask,
            pooling=pooling,
            require_grad=False,
        )

    def generate(self, ts_input, text_input_ids, max_new_tokens=128, **kwargs):
        """
        Generate thoughts (Reasoning) based on Time Series and Context.
        ts_input: [Batch, Dim, Time]
        text_input_ids: [Batch, Seq_Len] (The prompt)
        """
        _, _, inputs_embeds, full_attention_mask, _ = self._prepare_multimodal_inputs(
            ts_input,
            text_input_ids,
        )
        
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = self.llm.config.eos_token_id

        try:
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            return outputs
        except Exception as e:
            print(f"Warning: generate() failed with inputs_embeds: {e}")
            return torch.zeros((ts_input.shape[0], 1), dtype=torch.long, device=ts_input.device)


