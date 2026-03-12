"""
Shared lightweight tokenizer utilities.
"""
import torch


class DummyTokenizer:
    """
    A minimal tokenizer for offline/prototype testing when offline.
    Maps characters to basic IDs (ASCII + Offset).
    """

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "[EOS]"
        self.eos_token_id = 1
        self.unk_token = "[UNK]"
        self.unk_token_id = 2
        self.offset = 10

    def __call__(self, text, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        if isinstance(text, str):
            text = [text]

        batch_ids = []
        max_len = 0

        for t in text:
            ids = []
            for c in t:
                code = ord(c) + self.offset
                if code >= self.vocab_size:
                    code = self.unk_token_id
                ids.append(code)

            if truncation:
                ids = ids[:max_length]

            max_len = max(max_len, len(ids))
            batch_ids.append(ids)

        padded_ids = []
        attention_mask = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        if return_tensors == "pt":
            return type('BatchEncoding', (object,), {
                'input_ids': torch.tensor(padded_ids),
                'attention_mask': torch.tensor(attention_mask)
            })()

        return batch_ids

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid < self.offset:
                continue
            if tid == self.eos_token_id:
                break

            code = tid - self.offset
            if 0 <= code < 256:
                chars.append(chr(code))
            else:
                chars.append("?")

        return "".join(chars)