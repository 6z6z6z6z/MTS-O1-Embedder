"""
Data Generator for Cold Start / Teacher Phase
Responsibilities:
1. Load Raw Time Series.
2. Extract Statistical Features (Mean, Std, Trend, Peaks).
3. Generate teacher thoughts via DeepSeek API (label-blind) or fallback templates.
"""

import numpy as np
import os
import json
import random
import re
import time
import urllib.request
import urllib.error
from tqdm import tqdm
from mts_agent.data.adapters import ensure_channel_first, infer_dataset_name, load_dataset_by_adapter
from mts_agent.data.prompt_templates import build_fallback_reasoning, build_llm_prompts, get_context_template


# ─────────────────────── Dataset-specific prompt configs ─────────────────────

_NATOPS_CHANNELS = [
    'HandL_X','HandL_Y','HandL_Z','HandR_X','HandR_Y','HandR_Z',
    'ElbowL_X','ElbowL_Y','ElbowL_Z','ElbowR_X','ElbowR_Y','ElbowR_Z',
    'WristL_X','WristL_Y','WristL_Z','WristR_X','WristR_Y','WristR_Z',
    'ThumbL_X','ThumbL_Y','ThumbL_Z','ThumbR_X','ThumbR_Y','ThumbR_Z',
]

DATASET_PROMPT_CONFIGS = {
    "natops": {
        "system": "You are a motion analysis expert familiar with aviation hand signals and motion capture data.",
        "context": (
            "This is a motion capture recording of a pilot performing a standardized aviation hand gesture.\n"
            "Sensors: 8 locations (hand tips, elbows, wrists, thumbs) × 3D coordinates (X, Y, Z).\n"
            "Duration: ~1 second, 51 time steps. Values are normalized spatial coordinates.\n"
            "The 6 gesture classes are: 1=I have command, 2=All clear, 3=Not clear, "
            "4=Spread wings, 5=Fold wings, 6=Lock wings."
        ),
        "analysis_request": (
            "Analyze this motion recording. Focus on:\n"
            "1. Which body parts move the most? (check X/Y/Z ranges for hands, elbows, wrists)\n"
            "2. Is the motion symmetric (both sides move equally) or asymmetric (one side dominates)?\n"
            "3. What is the overall trajectory? (extending outward, contracting, pointing, crossing, etc.)\n"
            "Write 2-3 sentences describing the distinctive motion pattern. Do NOT guess or state the class label."
        ),
        "channel_names": _NATOPS_CHANNELS,
        "downsample": 3,   # 51 → 17 rows
    },
}


def _get_generic_prompt_config(dataset_name: str) -> dict:
    """Fall-back config for datasets without a specific entry."""
    return {
        "system": "You are an expert time-series analyst.",
        "context": f"This is a multivariate time-series sample from the '{dataset_name}' dataset.",
        "analysis_request": (
            "Analyze this time-series recording. Focus on:\n"
            "1. Which channels show the most variation?\n"
            "2. Are there any notable trends, peaks, or cross-channel patterns?\n"
            "Write 2-3 sentences describing distinctive features. Do NOT guess the class label."
        ),
        "channel_names": None,
        "downsample": 2,
    }


# ─────────────────────────── Table formatter ──────────────────────────────────

def format_ts_as_table(ts_data: np.ndarray, channel_names=None, downsample: int = 2) -> str:
    """Format (C, T) array as a compact pipe-delimited table."""
    ts = ensure_channel_first(ts_data)             # (C, T)
    ts_ds = ts[:, ::downsample]                    # downsample time axis
    n_ch, n_t = ts_ds.shape
    if channel_names and len(channel_names) == n_ch:
        header = "Time|" + "|".join(channel_names)
    else:
        header = "Time|" + "|".join(f"C{c}" for c in range(n_ch))
    lines = [header]
    for t in range(n_t):
        vals = "|".join(f"{ts_ds[c, t]:.2f}" for c in range(n_ch))
        lines.append(f"T{t * downsample}|{vals}")
    return "\n".join(lines)


# ───────────────────────── DeepSeek API caller ────────────────────────────────

def call_deepseek_api(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int = 200,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Call DeepSeek chat API. Returns the generated text or raises on failure."""
    url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429:                       # rate limit
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(f"DeepSeek HTTP {e.code}: {body}") from e
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(retry_delay)
                continue
            raise RuntimeError(f"DeepSeek call failed: {e}") from e
    raise RuntimeError("DeepSeek call failed after all retries.")


def generate_thought_deepseek(
    ts_data: np.ndarray,
    dataset_name: str,
    api_key: str,
) -> str:
    """Generate a label-blind thought for one time-series sample via DeepSeek."""
    key = dataset_name.lower()
    cfg = DATASET_PROMPT_CONFIGS.get(key, _get_generic_prompt_config(dataset_name))

    table = format_ts_as_table(
        ts_data,
        channel_names=cfg.get("channel_names"),
        downsample=cfg.get("downsample", 2),
    )

    user_prompt = (
        f"{cfg['context']}\n\n"
        f"Time-series data:\n{table}\n\n"
        f"{cfg['analysis_request']}"
    )

    return call_deepseek_api(
        system_prompt=cfg["system"],
        user_prompt=user_prompt,
        api_key=api_key,
    )


def format_time_series_as_table(ts_data):
    """
    Table Encoding Strategy (TableTime-inspired)
    Converts multivariate time series into a DFLoader-like text table.
    
    Structure:
    Timestep | Ch_0 | Ch_1 | ... | Ch_N
    0        | 0.12 | -0.5 | ... | 1.22
    ...
    """
    ts_data = ensure_channel_first(ts_data)

    # Downsample time by factor of 2 to save tokens (50 -> 25 rows)
    step_size = 2 
    ts_data = ts_data[:, ::step_size] 
    n_channels, n_steps = ts_data.shape
    
    # Header (Compact)
    header = "Time|" + "|".join([f"C{i}" for i in range(n_channels)])
    lines = [header]
    
    # Rows
    for t in range(n_steps):
        row_vals = [f"{ts_data[c, t]:.2f}" for c in range(n_channels)]
        row_str = f"T{t*step_size}|" + "|".join(row_vals)
        lines.append(row_str)
        
    return "\n".join(lines)

def extract_statistical_features(ts_data):
    """
    Extracts features for the prompt.
    MODIFIED: Minimalist mode. Only extracting raw data representation for TableTime approach.
    We strip away pre-computed stats like Mean/Std/FFT to force the LLM to look at the raw data.
    """
    # Ensure standard shape (D, L)
    ts_data = ensure_channel_first(ts_data)
    
    features = {}
    
    # Still keep these for the fallback template logic (if API fails), 
    # but we won't emphasize them in the main Prompt.
    features['mean'] = float(np.mean(ts_data))
    features['std'] = float(np.std(ts_data))
    
    # 3. Trend
    slope_sum = 0
    x = np.arange(ts_data.shape[1])
    for dim in range(ts_data.shape[0]):
        if ts_data.shape[1] > 1:
            slope, _ = np.polyfit(x, ts_data[dim], 1)
            slope_sum += slope
    
    avg_slope = slope_sum / ts_data.shape[0] if ts_data.shape[1] > 1 else 0
    
    if avg_slope > 0.02:
        features['trend'] = 'increasing'
    elif avg_slope < -0.02:
        features['trend'] = 'decreasing'
    else:
        features['trend'] = 'stable'

    features['volatility'] = "high" if features['std'] > 1.0 else "low"
    
    # Generic channel grouping summary
    half_dim = max(1, ts_data.shape[0] // 2)
    first_half_mean = float(np.mean(ts_data[:half_dim, :]))
    second_half_mean = float(np.mean(ts_data[half_dim:, :])) if ts_data.shape[0] > 1 else first_half_mean

    features['first_half_mean'] = f"{first_half_mean:.2f}"
    features['second_half_mean'] = f"{second_half_mean:.2f}"
    features['dominance'] = "first channel group" if first_half_mean > second_half_mean else "second channel group"
    features['num_channels'] = int(ts_data.shape[0])
    features['num_steps'] = int(ts_data.shape[1])
    
    # 4. TableTime Representation (Raw Data Table) - THE CORE FEATURE
    features['table_text'] = format_time_series_as_table(ts_data)

    return features

def sanitize_teacher_thought(thought_text, dataset_name, label=None):
    """Normalize teacher_thought into a generic, stable format."""
    label_text = str(label) if label is not None else "Unknown"

    def _truncate_words(text, max_words=22):
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip(' ,;:.') + "..."

    raw_lines = [line.strip() for line in thought_text.splitlines() if line.strip()]
    filtered_lines = []
    conclusion_line = None
    for line in raw_lines:
        lowered = line.lower()
        if any(bad in lowered for bad in ["i was given", "provided label", "hypothesis", "ground truth", "hidden supervision"]):
            continue
        line = re.sub(r'\*+', '', line)
        line = re.sub(r'^\d+\.\s*', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        if not line:
            continue
        if lowered.startswith("conclusion"):
            conclusion_line = f"Conclusion: {label_text}"
            continue
        if lowered.startswith("thinking process"):
            continue
        filtered_lines.append(line)

    compact_lines = []
    if filtered_lines:
        compact_lines.append(f"Overview: {_truncate_words(filtered_lines[0])}")
    if len(filtered_lines) > 1:
        compact_lines.append(f"Evidence: {_truncate_words(filtered_lines[1])}")
    if len(filtered_lines) > 2:
        compact_lines.append(f"Pattern: {_truncate_words(filtered_lines[2])}")

    if conclusion_line is None:
        conclusion_line = f"Conclusion: {label_text}"
    compact_lines.append(conclusion_line)

    text = "\n".join(compact_lines).strip()
    if not text:
        return f"Conclusion: {label_text}"
    return text

def generate_text_from_template_fallback(features, dataset_name, label=None):
    """
    Fallback Template Generator (Same as original).
    """
    
    context = get_context_template(dataset_name)
    reasoning_logic = build_fallback_reasoning(features, dataset_name, label)

    thought_template = (
        f"Thinking Process:\n"
        f"1.  **Statistical Overview**: The signal has a mean of {features['mean']:.2f} and a standard deviation of {features['std']:.2f}, "
        f"suggesting {features['volatility']} volatility.\n"
        f"2.  **Trend Analysis**: The aggregated trend appears {features['trend']} across the window.\n"
        f"{reasoning_logic}"
    )

    thought_template = sanitize_teacher_thought(thought_template, dataset_name, label)

    return context, thought_template

def generate_thought_data(dataset_path: str, output_path: str, dataset_name="General_TS", api_key=None, max_samples=None):
    """
    Load a raw dataset and generate teacher thoughts.

    If api_key is provided, calls DeepSeek API for label-blind thought generation
    (the O1-Embedder approach).  Falls back to template-based generation otherwise.

    Supports .npy files or folders containing X_train.npy / y_train.npy.
    """
    use_deepseek = bool(api_key)
    print(f"Loading data from {dataset_path}...")
    if use_deepseek:
        print(f"DeepSeek API enabled — generating label-blind thoughts via API.")

    data = None
    labels = None

    dataset_name = infer_dataset_name(dataset_path, dataset_name)

    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        is_train = "train" in output_path.lower()
        split_mode = 'train' if is_train else 'valid'
        try:
            data, labels = load_dataset_by_adapter(dataset_path, mode=split_mode, dataset_name=dataset_name)
            print(f"Loaded {len(data)} samples from folder {dataset_path}")
        except Exception as e:
            print(f"Error loading folder dataset: {e}")
            return
    elif os.path.exists(dataset_path) and os.path.isfile(dataset_path):
        try:
            data, labels = load_dataset_by_adapter(dataset_path, mode='train', dataset_name=dataset_name)
            if labels is None:
                labels = [None] * len(data)
            print(f"Loaded {len(data)} samples from file {dataset_path}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    
    # Check data
    if data is None:
        print("Error: No valid data loaded.")
        return

    generated_samples = []
    
    # Subsampling if requested
    total_samples = len(data)
    indices = range(total_samples)
    if max_samples and max_samples < total_samples:
        indices = random.sample(indices, max_samples)
        indices.sort()
        
    # Save to JSONL progressively
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mode = "DeepSeek API" if use_deepseek else "template fallback"
    print(f"Generating thoughts for {len(indices)} samples [{mode}] → {output_path}...")

    failed = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in tqdm(indices):
            sample_ts = ensure_channel_first(data[i])
            label = labels[i] if labels is not None else None

            # 1. Context string (same regardless of thought source)
            context = get_context_template(dataset_name)

            # 2. Thought generation
            if use_deepseek:
                try:
                    thought = generate_thought_deepseek(sample_ts, dataset_name, api_key)
                    # Small delay to respect rate limits
                    time.sleep(0.5)
                except Exception as e:
                    print(f"\n  [Warning] DeepSeek failed for sample {i}: {e}. Using template fallback.")
                    stats = extract_statistical_features(sample_ts)
                    _, thought = generate_text_from_template_fallback(stats, dataset_name, label)
                    failed += 1
            else:
                stats = extract_statistical_features(sample_ts)
                _, thought = generate_text_from_template_fallback(stats, dataset_name, label)

            # 3. Structuring
            record = {
                "id": int(i),
                "time_series": sample_ts.tolist(),
                "label": str(label) if label is not None else "Unknown",
                "context": context,
                "teacher_thought": thought,
                "thought": thought,
                "thought_source": "deepseek" if use_deepseek else "template",
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

    if failed:
        print(f"  {failed}/{len(indices)} samples fell back to template (DeepSeek errors).")
    print(f"Finished generation. Output at: {output_path}")
