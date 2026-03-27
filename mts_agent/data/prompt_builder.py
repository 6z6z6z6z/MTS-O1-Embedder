"""
Shared prompt building utilities for training, inference and retrieval.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional


SYSTEM_PROMPT = "You are a time series analyst. Analyze the provided data trend and context."
VISION_PLACEHOLDER = "<|vision_start|><|vision_pad|><|vision_end|>"
# TS tokens placed at the END of the prompt so that:
#   (a) TS tokens can attend to the full task instruction (causal attention flows left→right)
#   (b) the last TS token (vision_end) serves as a natural EOS-style pooling anchor
#       and has seen all preceding text when used with `embedding_pooling="last"`
USER_TEMPLATE = "Context: {context}\nTask: Analyze patterns.\nData: {vision_placeholder}"
RESPONSE_PREFIX = "Response:"
VALID_ALIGNMENT_TEXT_MODES = frozenset({"full", "context", "label", "context_label"})

# Human-readable class names for known datasets (used in context_label mode).
_NATOPS_LABEL_NAMES: dict[str, str] = {
    "1": "I have command", "1.0": "I have command",
    "2": "All clear",      "2.0": "All clear",
    "3": "Not clear",      "3.0": "Not clear",
    "4": "Spread wings",   "4.0": "Spread wings",
    "5": "Fold wings",     "5.0": "Fold wings",
    "6": "Lock wings",     "6.0": "Lock wings",
}


def _resolve_label_name(item: Mapping[str, object], raw_label: str) -> str:
    """Return a human-readable label name when available.

    Priority:
    1. Explicit ``label_name`` field in the item.
    2. NATOPS built-in mapping (based on numeric label string).
    3. Raw label string as fallback.
    """
    explicit = str(item.get('label_name') or '').strip()
    if explicit:
        return explicit
    return _NATOPS_LABEL_NAMES.get(raw_label, raw_label)


def normalize_context(context: Optional[str]) -> str:
    return str(context or "").strip()


def _append_response(prompt: str, thought: Optional[str], include_response_stub: bool) -> str:
    cleaned_thought = str(thought or "").strip()
    if cleaned_thought:
        return f"{prompt}\n{RESPONSE_PREFIX} {cleaned_thought}"
    if include_response_stub:
        return f"{prompt}\n{RESPONSE_PREFIX}"
    return prompt


def _validate_alignment_text_mode(alignment_text_mode: str) -> str:
    normalized_mode = str(alignment_text_mode or "").strip().lower()
    if normalized_mode not in VALID_ALIGNMENT_TEXT_MODES:
        valid_modes = ", ".join(sorted(VALID_ALIGNMENT_TEXT_MODES))
        raise ValueError(f"Unsupported alignment_text_mode '{alignment_text_mode}'. Expected one of: {valid_modes}")
    return normalized_mode


def _build_context_only_alignment_text(context: str, label_display: str) -> str:
    if label_display:
        return f"Context: {context}\nClass label: {label_display}."
    return f"Context: {context}"


def build_user_prompt(context: Optional[str]) -> str:
    return USER_TEMPLATE.format(
        context=normalize_context(context),
        vision_placeholder=VISION_PLACEHOLDER
    )


def build_full_prompt(context: Optional[str], thought: Optional[str] = None, include_response_stub: bool = True) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(context)}"
    return _append_response(prompt, thought, include_response_stub)


def get_teacher_thought(item: Mapping[str, object]) -> str:
    return str(item.get("teacher_thought") or item.get("thought") or "").strip()


def build_alignment_text(item: Mapping[str, object], alignment_text_mode: str = "full", full_text: Optional[str] = None) -> str:
    context = normalize_context(item.get('context'))
    label = str(item.get('label') or '').strip()
    normalized_mode = _validate_alignment_text_mode(alignment_text_mode)

    builders: dict[str, Callable[[], str]] = {
        "full": lambda: full_text if full_text is not None else build_full_prompt(context, get_teacher_thought(item), include_response_stub=False),
        "context": lambda: f"Context summary: {context}",
        "label": lambda: f"Time-series class label: {label}." if label else "Time-series class label unknown.",
        "context_label": lambda: _build_context_only_alignment_text(context, _resolve_label_name(item, label)),
    }
    return builders[normalized_mode]()


def build_retrieval_prompt(context: Optional[str]) -> str:
    return build_full_prompt(context, thought=None, include_response_stub=False)
