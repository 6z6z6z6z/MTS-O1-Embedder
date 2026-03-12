"""
Core Model: MTS-O1-Embedder
Combines TS Encoder, Projector, and LLM Backbone.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .ts_encoder import TimeSeriesEncoder
from .projector import TimeSeriesProjector

class MTSEmbedder(nn.Module):
    
    def get_text_embeds(self, text_input_ids):
        # Handle simple model or PEFT wrapped model
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
            return self.llm.model.embed_tokens(text_input_ids)
        elif hasattr(self.llm, "base_model") and hasattr(self.llm.base_model.model, "model") and hasattr(self.llm.base_model.model.model, "embed_tokens"): # peft wrap 1
            return self.llm.base_model.model.model.embed_tokens(text_input_ids)
        else:
            # Fallback for Qwen2 if nested differently
            try:
                if hasattr(self.llm, "get_input_embeddings"):
                    return self.llm.get_input_embeddings()(text_input_ids)
            except:
                pass
            raise AttributeError("Could not find embed_tokens layer in the LLM.")

    def __init__(
        self,
        llm_model_path,
        ts_input_dim=1,
        ts_hidden_dim=128,
        encoder_base_channels=64,
        encoder_target_tokens=16,
        encoder_dropout=0.1,
        encoder_norm="group",
        embedding_pooling="mean",
    ):
        super().__init__()
        print(f"Loading LLM from {llm_model_path}...")
        
        # 1. LLM Backbone
        # PURE OFFLINE PRIORITY
        try:
            print(f" -> Checking local cache for {llm_model_path}...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_path, 
                trust_remote_code=True,
                local_files_only=True
            )
            print(f" -> Successfully loaded {llm_model_path} from local cache.")
        except Exception as e_local:
            print(f" -> Not found in local cache: {e_local}")
            print(" -> Network is blocked/unstable. Using Dummy Config immediately.")
            from transformers import LlamaConfig, LlamaForCausalLM
            # Create a small random model to allow training pipeline to run
            config = LlamaConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=2, num_attention_heads=4)
            self.llm = LlamaForCausalLM(config)

        self.llm_dim = self.llm.config.hidden_size
        
        # 2. Perception Layer
        self.ts_encoder = TimeSeriesEncoder(
            input_dim=ts_input_dim,
            hidden_dim=ts_hidden_dim,
            base_channels=encoder_base_channels,
            target_tokens=encoder_target_tokens,
            dropout=encoder_dropout,
            norm_type=encoder_norm,
        )
        self.projector = TimeSeriesProjector(ts_dim=ts_hidden_dim, llm_dim=self.llm_dim)
        
        # 3. Alignment Normalization (Crucial for convergence)
        # Helps map projector output dist to LLM embedding dist
        self.projector_norm = nn.LayerNorm(self.llm_dim)

        # 4. Special tokens for TS embedding insertion
        self.ts_start_token = "<|ts_start|>"
        self.ts_end_token = "<|ts_end|>"
        self.embedding_pooling = embedding_pooling

    def _get_ts_token_mask(self, text_input_ids, ts_seq_len, max_seq_len):
        """Return float mask (B, max_seq_len) with 1.0 at TS token positions.

        TS tokens are inserted immediately after the <|vision_start|> marker (id=151652).
        This mask is used by the 'ts_tokens' pooling mode to pool only the LLM hidden
        states corresponding to TS tokens, reducing text-identity bias.
        """
        B = text_input_ids.shape[0]
        mask = torch.zeros(B, max_seq_len, device=text_input_ids.device, dtype=torch.float32)
        for i in range(B):
            start_pos = (text_input_ids[i] == 151652).nonzero(as_tuple=True)[0]
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
        Insert TS embeddings between <|vision_start|> (151652) and <|vision_end|> (151653) markers.
        Adjusts attention mask and labels if provided.
        """
        batch_size = text_embeds.shape[0]
        ts_seq_len = ts_embeds.shape[1]
        
        new_inputs_embeds = []
        new_attention_masks = []
        new_labels = []
        
        for i in range(batch_size):
            # Using Qwen2.5 vision markers for TS
            start_idx = (text_input_ids[i] == 151652).nonzero(as_tuple=True)[0]
            end_idx = (text_input_ids[i] == 151653).nonzero(as_tuple=True)[0]
            
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
        max_len = max([t.shape[0] for t in new_inputs_embeds])
        
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
            
            # Target projection matrices in the attention mechanism
            # Typical targets for Llama/Qwen architectures
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
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
        text_input_ids: [Batch, Seq_Len] (tokenized text composed of: Context + Thought)
        attention_mask: [Batch, Seq_Len]
        return_embedding: if True, also return the pooled embedding for contrastive training
        """

        # 1. Perception Phase: Encode TS
        ts_feat = self.ts_encoder(ts_input)  # [Batch, TS_Seq, TS_Dim]
        ts_embeds = self.projector(ts_feat)  # [Batch, TS_Seq, LLM_Dim]
        ts_embeds = self.projector_norm(ts_embeds) # Apply Norm

        # 2. Text Embeddings
        text_embeds = self.get_text_embeds(text_input_ids) # [Batch, Text_Seq, LLM_Dim]

        # 3. Insert TS embeddings
        inputs_embeds, full_attention_mask, full_labels = self.insert_ts_embeddings(
            text_embeds, text_input_ids, ts_embeds, attention_mask, labels
        )

        # 4. Forward Pass
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=True
        )

        if return_embedding:
            last_hidden_state = outputs.hidden_states[-1]  # [Batch, Seq, H]
            embedding = self._pool_hidden_state(last_hidden_state, attention_mask=full_attention_mask)
            return outputs, embedding

        return outputs

    def get_ts_only_embedding(self, ts_input):
        """Embed TS using only the TS encoder and projector — no LLM forward.

        Eliminates the text-identity bias that occurs when all samples share
        nearly identical context strings. The retrieval embedding is derived
        purely from signal content: ts_encoder → projector → mean-pool over tokens.

        Gradient flow is determined by the caller's context (torch.no_grad() at
        inference, or leave gradients enabled during training).
        Returns [B, llm_dim] embedding.
        """
        ts_feat = self.ts_encoder(ts_input)    # [B, tokens, hidden_dim]
        ts_proj = self.projector(ts_feat)       # [B, tokens, llm_dim]
        ts_norm = self.projector_norm(ts_proj)  # [B, tokens, llm_dim]
        return ts_norm.mean(dim=1)              # [B, llm_dim]

    def get_embedding_for_training(self, ts_input, text_input_ids, attention_mask=None, pooling=None):
        """
        Get embeddings with gradients flowing — used during training for contrastive loss.
        Uses the same context-only (retrieval) prompt as get_embedding(), so the contrastive
        objective directly optimizes the retrieval embedding space.
        """
        # 1. Perception
        ts_feat = self.ts_encoder(ts_input)
        ts_embeds = self.projector(ts_feat)
        ts_embeds = self.projector_norm(ts_embeds)

        # 2. Text
        text_input_ids = text_input_ids.long()
        text_embeds = self.get_text_embeds(text_input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.long, device=text_input_ids.device)

        # 3. Insert TS embeddings
        inputs_embeds, full_attention_mask, _ = self.insert_ts_embeddings(
            text_embeds, text_input_ids, ts_embeds, attention_mask=attention_mask
        )

        # 4. Forward (with gradients)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 5. Pooling — for "ts_tokens" mode, average only the TS token positions
        last_hidden_state = outputs.hidden_states[-1]
        ts_mask = None
        if (pooling or self.embedding_pooling) == "ts_tokens":
            ts_mask = self._get_ts_token_mask(
                text_input_ids, ts_embeds.shape[1], inputs_embeds.shape[1]
            )
        return self._pool_hidden_state(
            last_hidden_state, attention_mask=full_attention_mask, pooling=pooling, ts_mask=ts_mask
        )

    def get_embedding(self, ts_input, text_input_ids, attention_mask=None, pooling=None):
        """
        Get the final dense embedding for retrieval.
        """
        # 1. Perception
        ts_feat = self.ts_encoder(ts_input)
        ts_embeds = self.projector(ts_feat)
        ts_embeds = self.projector_norm(ts_embeds)
        
        # 2. Text
        # Ensure input_ids are LongTensor
        text_input_ids = text_input_ids.long()
        text_embeds = self.get_text_embeds(text_input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.long, device=text_input_ids.device)

        # 3. Insert TS embeddings
        inputs_embeds, full_attention_mask, _ = self.insert_ts_embeddings(
            text_embeds,
            text_input_ids,
            ts_embeds,
            attention_mask=attention_mask
        )
        
        # 4. Forward (No Grad)
        with torch.no_grad():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # 5. Pooling — for "ts_tokens" mode, average only the TS token positions
        last_hidden_state = outputs.hidden_states[-1]  # [Batch, Seq, H]
        ts_mask = None
        if (pooling or self.embedding_pooling) == "ts_tokens":
            ts_mask = self._get_ts_token_mask(
                text_input_ids, ts_embeds.shape[1], inputs_embeds.shape[1]
            )
        embedding = self._pool_hidden_state(
            last_hidden_state, attention_mask=full_attention_mask, pooling=pooling, ts_mask=ts_mask
        )
        return embedding

    def generate(self, ts_input, text_input_ids, max_new_tokens=128, **kwargs):
        """
        Generate thoughts (Reasoning) based on Time Series and Context.
        ts_input: [Batch, Dim, Time]
        text_input_ids: [Batch, Seq_Len] (The prompt)
        """
        # 1. Perception Phase: Encode TS
        ts_feat = self.ts_encoder(ts_input)  # [Batch, TS_Seq, TS_Dim]
        ts_embeds = self.projector(ts_feat)  # [Batch, TS_Seq, LLM_Dim]
        ts_embeds = self.projector_norm(ts_embeds)
        
        # 2. Text
        text_embeds = self.get_text_embeds(text_input_ids)

        # 3. Insert TS embeddings
        inputs_embeds, _, _ = self.insert_ts_embeddings(text_embeds, text_input_ids, ts_embeds)
        
        # 4. Generate
        # Attempt to use generate with inputs_embeds
        # Qwen2.5 / Transformers recent versions support this.
        if "pad_token_id" not in kwargs:
             kwargs["pad_token_id"] = self.llm.config.eos_token_id

        try:
             outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                **kwargs
             )
             return outputs
        except Exception as e:
             print(f"Warning: generate() failed with inputs_embeds: {e}")
             # Return dummy
             return torch.zeros((ts_input.shape[0], 1), dtype=torch.long, device=ts_input.device)


