"""
Shared prompt building utilities for training, inference and retrieval.
"""
from __future__ import annotations

from typing import Mapping, Optional


SYSTEM_PROMPT = "You are a time series analyst. Analyze the provided data trend and context."
VISION_PLACEHOLDER = "<|vision_start|><|vision_pad|><|vision_end|>"
USER_TEMPLATE = "Context: {context}\nData: {vision_placeholder}\nTask: Analyze patterns."
RESPONSE_PREFIX = "Response:"


def normalize_context(context: Optional[str]) -> str:
    return str(context or "").strip()


def build_user_prompt(context: Optional[str]) -> str:
    return USER_TEMPLATE.format(
        context=normalize_context(context),
        vision_placeholder=VISION_PLACEHOLDER
    )


def build_full_prompt(context: Optional[str], thought: Optional[str] = None, include_response_stub: bool = True) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(context)}"
    cleaned_thought = str(thought or "").strip()
    if cleaned_thought:
        return f"{prompt}\n{RESPONSE_PREFIX} {cleaned_thought}"
    if include_response_stub:
        return f"{prompt}\n{RESPONSE_PREFIX}"
    return prompt


def get_teacher_thought(item: Mapping[str, object]) -> str:
    return str(item.get("teacher_thought") or item.get("thought") or "").strip()


def build_alignment_text(item: Mapping[str, object], alignment_text_mode: str = "full", full_text: Optional[str] = None) -> str:
    context = normalize_context(item.get('context'))
    label = str(item.get('label') or '').strip()

    if alignment_text_mode == "full":
        return full_text if full_text is not None else build_full_prompt(context, get_teacher_thought(item), include_response_stub=False)
    if alignment_text_mode == "label":
        return f"Time-series class label: {label}." if label else "Time-series class label unknown."
    if alignment_text_mode == "context_label":
        if label:
            return f"Context: {context}\nClass label: {label}."
        return f"Context: {context}"
    return f"Context summary: {context}"


def build_retrieval_prompt(context: Optional[str]) -> str:
    return build_full_prompt(context, thought=None, include_response_stub=False)
