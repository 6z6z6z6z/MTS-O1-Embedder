"""
Data Collator for Multimodal Inputs
Responsibilities:
1. Pad Time Series sequences to the longest in the batch.
2. Construct the Conversation Prompt (Context + <TS> + Thought).
3. Tokenize text inputs.
"""
import torch

from mts_agent.data.prompt_builder import build_alignment_text, build_full_prompt, get_teacher_thought, RESPONSE_PREFIX, build_retrieval_prompt

class MultimodalCollator:
    def __init__(self, tokenizer, max_length=1024, mode="train", alignment_text_mode="full", include_debug_text=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode # 'train' includes thoughts in labels, 'inference' does not.
        self.alignment_text_mode = alignment_text_mode
        self.include_debug_text = include_debug_text
        # Cache "Response:" token IDs for prompt masking (computed once)
        try:
            self._response_token_ids = tokenizer.encode(RESPONSE_PREFIX, add_special_tokens=False)
        except Exception:
            self._response_token_ids = []

    def build_full_text(self, item):
        if self.mode == "train":
            return build_full_prompt(item.get('context'), get_teacher_thought(item), include_response_stub=False)
        return build_full_prompt(item.get('context'), thought=None, include_response_stub=True)

    def build_alignment_text(self, item, full_text):
        return build_alignment_text(item, alignment_text_mode=self.alignment_text_mode, full_text=full_text)

    def _normalize_time_series(self, ts):
        if not torch.is_tensor(ts):
            ts = torch.as_tensor(ts)
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        elif ts.ndim == 2 and ts.shape[0] > ts.shape[1]:
            ts = ts.transpose(0, 1)
        elif ts.ndim != 2:
            raise ValueError(f"time_series must be 1D or 2D, got shape={tuple(ts.shape)}")
        return ts

    def _pad_time_series_batch(self, batch, key='time_series'):
        ts_list = [self._normalize_time_series(item[key]) for item in batch]
        max_channels = max(ts.shape[0] for ts in ts_list)
        max_time = max(ts.shape[1] for ts in ts_list)
        ts_batch = torch.zeros(len(ts_list), max_channels, max_time, dtype=ts_list[0].dtype)
        for i, ts in enumerate(ts_list):
            channels, time_steps = ts.shape
            ts_batch[i, :channels, :time_steps] = ts
        return ts_batch

    def _tokenize_texts(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def _build_lm_labels(self, input_ids):
        if self.mode != "train":
            return None

        lm_labels = input_ids.clone()
        if not self._response_token_ids:
            return lm_labels

        response_len = len(self._response_token_ids)
        for batch_idx in range(lm_labels.size(0)):
            ids = input_ids[batch_idx].tolist()
            mask_up_to = None
            for token_idx in range(len(ids) - response_len + 1):
                if ids[token_idx:token_idx + response_len] == self._response_token_ids:
                    mask_up_to = token_idx + response_len
                    break
            if mask_up_to is not None:
                lm_labels[batch_idx, :mask_up_to] = -100
            else:
                lm_labels[batch_idx, :] = -100
        return lm_labels

    def _build_text_variants(self, batch):
        text_inputs = []
        alignment_texts = []
        retrieval_texts = []
        for item in batch:
            full_text = self.build_full_text(item)
            text_inputs.append(full_text)
            alignment_texts.append(self.build_alignment_text(item, full_text))
            retrieval_texts.append(build_retrieval_prompt(item.get('context')))
        return text_inputs, alignment_texts, retrieval_texts
        
    def __call__(self, batch):
        # batch is a list of dicts from MTSDataset
        if not batch:
            raise ValueError("MultimodalCollator received an empty batch.")
        
        # 1. Process Time Series (view1 = query, view2 = gallery safe-aug)
        ts_batch = self._pad_time_series_batch(batch)
        if 'time_series_view2' in batch[0]:
            ts_batch_view2 = self._pad_time_series_batch(batch, key='time_series_view2')
        else:
            ts_batch_view2 = ts_batch
        
        # 2. Process Text
        # We need to construct the full text for the LLM.
        # Format: [System] [User info...] [Thought (if train)]
        text_inputs, alignment_texts, retrieval_texts = self._build_text_variants(batch)
            
        # Tokenize (Padding & Truncation)
        tokenized = self._tokenize_texts(text_inputs)
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        alignment_tokenized = self._tokenize_texts(alignment_texts)
        alignment_input_ids = alignment_tokenized.input_ids

        retrieval_tokenized = self._tokenize_texts(retrieval_texts)
        retrieval_input_ids = retrieval_tokenized.input_ids
        retrieval_attention_mask = retrieval_tokenized.attention_mask
        
        # 3. Create Labels (for Causal LM training)
        lm_labels = self._build_lm_labels(input_ids)
        
        # 4. Extract Classification Labels & IDs (Preserve Metadata)
        class_labels = [item['label'] for item in batch]
        ids = [item['id'] for item in batch]

        output = {
            "ts_input": ts_batch,            # [Batch, Dim, Time] — view1 (heavy aug), for query
            "ts_input_view2": ts_batch_view2, # [Batch, Dim, Time] — view2 (safe aug), for gallery
            "text_input_ids": input_ids,     # [Batch, Seq_Len]
            "alignment_input_ids": alignment_input_ids,
            "retrieval_input_ids": retrieval_input_ids,       # context-only, for contrastive
            "retrieval_attention_mask": retrieval_attention_mask,
            "attention_mask": attention_mask,# [Batch, Seq_Len]
            "labels": lm_labels,             # [Batch, Seq_Len] (Token Labels for Next Token Prediction)
            "label": class_labels,           # [Batch] (Classification Labels for Evaluation)
            "id": ids,                       # [Batch] (Sample IDs)
        }

        if self.include_debug_text:
            output["raw_text"] = text_inputs
            output["raw_alignment_text"] = alignment_texts

        return output

