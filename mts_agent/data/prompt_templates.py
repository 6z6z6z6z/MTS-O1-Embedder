"""
Prompt/template registry for generic time-series reasoning.

Built-in: one generic template applicable to any dataset.
Extension: register dataset-specific templates via register_prompt_template() or the
@prompt_template() decorator — no changes to core training code required.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple


ContextBuilder = Callable[[str], List[str]]
ReasoningBuilder = Callable[[dict, str], str]


# ──────────────────────────── Generic (default) ────────────────────────────

def _generic_contexts(dataset_name: str) -> List[str]:
    return [
        f"This dataset '{dataset_name}' contains multivariate time-series observations collected over time.",
        f"Data recorded from the '{dataset_name}' domain, monitoring one or more variables across a sequence.",
        f"The following sample is a multivariate time-series segment extracted from '{dataset_name}'.",
    ]


def _generic_reasoning(features: dict, label: str) -> str:
    return (
        f"3.  **Cross-Channel Summary**: The sequence contains {features['num_channels']} channel(s) across "
        f"{features['num_steps']} steps, and the {features['dominance']} is slightly more active.\n"
        f"4.  **Conclusion**: General pattern analysis suggests class '{label}'."
    )


# ───────────────────────────── Registry API ────────────────────────────────

def _natops_contexts(dataset_name: str) -> List[str]:
    return [
        "This is a motion capture recording of a pilot performing a standardized aviation hand gesture. "
        "Sensors are attached to 8 body locations (hand tips, elbows, wrists, thumbs) recording X/Y/Z coordinates.",
        "24-channel 3D motion capture data of a pilot's hand gesture signal used in aviation communication. "
        "Channels encode spatial coordinates for left/right hand tips, elbows, wrists, and thumbs.",
    ]


def _natops_reasoning(features: dict, label: str) -> str:
    class_names = {
        "1.0": "I have command", "1": "I have command",
        "2.0": "All clear",      "2": "All clear",
        "3.0": "Not clear",      "3": "Not clear",
        "4.0": "Spread wings",   "4": "Spread wings",
        "5.0": "Fold wings",     "5": "Fold wings",
        "6.0": "Lock wings",     "6": "Lock wings",
    }
    name = class_names.get(str(label), str(label))
    return (
        f"3.  **Motion Pattern**: The gesture corresponds to '{name}', characterised by "
        f"{'bilateral symmetric arm motion' if str(label) in ('4.0','5.0','4','5') else 'asymmetric single-arm motion'}.\n"
        f"4.  **Conclusion**: The motion pattern is consistent with gesture class '{label}' ({name})."
    )


TEMPLATE_REGISTRY: Dict[str, Tuple[ContextBuilder, ReasoningBuilder]] = {
    "generic": (_generic_contexts, _generic_reasoning),
    "natops":  (_natops_contexts,  _natops_reasoning),
}


def register_prompt_template(key: str, context_builder: ContextBuilder, reasoning_builder: ReasoningBuilder):
    """Register or override a prompt template pair by dataset key.

    Example::

        def my_contexts(dataset_name):
            return [f"ECG recording from {dataset_name}."]

        def my_reasoning(features, label):
            return f"Signal class: {label}."

        register_prompt_template("my_ecg", my_contexts, my_reasoning)
    """
    TEMPLATE_REGISTRY[key.lower()] = (context_builder, reasoning_builder)


def prompt_template(key: str, context_builder: ContextBuilder):
    """Decorator helper for registering a reasoning builder together with a context builder.

    Example::

        @prompt_template("my_ecg", my_contexts)
        def my_reasoning(features, label):
            return f"Signal class: {label}."
    """
    def decorator(reasoning_builder: ReasoningBuilder):
        register_prompt_template(key, context_builder, reasoning_builder)
        return reasoning_builder
    return decorator


def resolve_template_key(dataset_name: str) -> str:
    """Find the best-matching registered template key for *dataset_name*.

    Performs case-insensitive substring matching against registered keys.
    Falls back to 'generic' when no match is found.
    """
    lowered = (dataset_name or "").lower()
    # Exact match first
    if lowered in TEMPLATE_REGISTRY:
        return lowered
    # Substring match (skip 'generic' to avoid false positives)
    for key in TEMPLATE_REGISTRY:
        if key != "generic" and key in lowered:
            return key
    return "generic"


def get_context_template(dataset_name: str) -> str:
    key = resolve_template_key(dataset_name)
    context_builder, _ = TEMPLATE_REGISTRY[key]
    return random.choice(context_builder(dataset_name))


def build_fallback_reasoning(features: dict, dataset_name: str, label: str) -> str:
    key = resolve_template_key(dataset_name)
    _, reasoning_builder = TEMPLATE_REGISTRY[key]
    return reasoning_builder(features, label)


def build_llm_prompts(features: dict, dataset_name: str, label: str) -> Tuple[str, str, str]:
    """Build (context, system_prompt, user_prompt) for teacher-thought generation."""
    context = get_context_template(dataset_name)
    system_prompt = (
        "You are an expert time-series analyst. "
        "Your task is to analyze a time-series sample and provide a concise chain-of-thought style rationale. "
        "You must conclude the classification label clearly."
    )
    user_prompt = (
        f"Domain: {dataset_name}\n"
        f"Context: {context}\n"
        f"Raw Data Table (Downsampled):\n"
        f"{features['table_text']}\n\n"
        f"Hypothesis: The signal matches class '{label}'.\n\n"
        "Task: Confirm this hypothesis by analyzing the raw data table directly.\n"
        "1.  **Scan the Table**: Look at channel-wise values across time.\n"
        "2.  **Find the Pattern**: Identify channels or segments that show distinct behaviour supporting the label.\n"
        "3.  **Explain**: Connect those specific values to the conclusion.\n"
        "CRITICAL: Do NOT mention you were given the label.\n"
        "Output format:\nThinking Process: ...\nConclusion: {label}"
    )
    return context, system_prompt, user_prompt

