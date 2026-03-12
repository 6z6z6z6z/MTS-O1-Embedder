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
    def __init__(self, tokenizer, max_length=1024, mode="train", alignment_text_mode="full"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode # 'train' includes thoughts in labels, 'inference' does not.
        self.alignment_text_mode = alignment_text_mode
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
        
    def __call__(self, batch):
        # batch is a list of dicts from MTSDataset
        
        # 1. Process Time Series
        # Find max seq length in this batch for padding
        ts_list = []
        for item in batch:
            ts = item['time_series']
            if ts.ndim == 1:
                ts = ts.unsqueeze(0)
            elif ts.ndim == 2 and ts.shape[0] > ts.shape[1]:
                ts = ts.transpose(0, 1)
            ts_list.append(ts) # [Dim, Time]

        max_channels = max(ts.shape[0] for ts in ts_list)
        max_time = max(ts.shape[1] for ts in ts_list)
        ts_batch = torch.zeros(len(ts_list), max_channels, max_time, dtype=ts_list[0].dtype)
        for i, ts in enumerate(ts_list):
            c, t = ts.shape
            ts_batch[i, :c, :t] = ts
        
        # 2. Process Text
        # We need to construct the full text for the LLM.
        # Format: [System] [User info...] [Thought (if train)]
        
        text_inputs = []
        alignment_texts = []
        retrieval_texts = []
        for item in batch:
            full_text = self.build_full_text(item)
            text_inputs.append(full_text)
            alignment_texts.append(self.build_alignment_text(item, full_text))
            retrieval_texts.append(build_retrieval_prompt(item.get('context')))
            
        # Tokenize (Padding & Truncation)
        tokenized = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        alignment_tokenized = self.tokenizer(
            alignment_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        alignment_input_ids = alignment_tokenized.input_ids

        retrieval_tokenized = self.tokenizer(
            retrieval_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        retrieval_input_ids = retrieval_tokenized.input_ids
        retrieval_attention_mask = retrieval_tokenized.attention_mask
        
        # 3. Create Labels (for Causal LM training)
        lm_labels = None
        if self.mode == "train":
            lm_labels = input_ids.clone()
            # Prompt masking: only compute LM loss on teacher_thought tokens.
            # Find "Response:" in each sequence and mask all tokens up to and including it.
            if self._response_token_ids:
                n_resp = len(self._response_token_ids)
                for b in range(lm_labels.size(0)):
                    ids_b = input_ids[b].tolist()
                    mask_up_to = None
                    for j in range(len(ids_b) - n_resp + 1):
                        if ids_b[j:j + n_resp] == self._response_token_ids:
                            mask_up_to = j + n_resp  # include "Response:" itself
                            break
                    if mask_up_to is not None:
                        lm_labels[b, :mask_up_to] = -100
                    else:
                        # "Response:" not found — mask entire sequence to avoid spurious loss
                        lm_labels[b, :] = -100
        
        # 4. Extract Classification Labels & IDs (Preserve Metadata)
        class_labels = [item['label'] for item in batch]
        ids = [item['id'] for item in batch]
        
        return {
            "ts_input": ts_batch,            # [Batch, Dim, Time]
            "text_input_ids": input_ids,     # [Batch, Seq_Len]
            "alignment_input_ids": alignment_input_ids,
            "retrieval_input_ids": retrieval_input_ids,       # context-only, for contrastive
            "retrieval_attention_mask": retrieval_attention_mask,
            "attention_mask": attention_mask,# [Batch, Seq_Len]
            "labels": lm_labels,             # [Batch, Seq_Len] (Token Labels for Next Token Prediction)
            "label": class_labels,           # [Batch] (Classification Labels for Evaluation)
            "id": ids,                       # [Batch] (Sample IDs)
            "raw_text": text_inputs,         # For debugging
            "raw_alignment_text": alignment_texts
        }

