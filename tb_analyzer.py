#!/usr/bin/env python
"""CLI to convert a TensorBoard trial directory into an LLM-friendly markdown report.

Usage:
    python tb_analyzer.py <path_to_trial_tfboard_dir>

Emits markdown to stdout with three sections:
    1. Text summaries (hyperparameters)
    2. Scalars under each tag namespace (e.g. train/, eval/)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    TENSORS,
    SCALARS,
)
from tensorboard.util import tensor_util


def load_accumulator(path: str) -> EventAccumulator:
    # size_guidance=0 means load everything (no downsampling)
    acc = EventAccumulator(
        path,
        size_guidance={SCALARS: 0, TENSORS: 0},
    )
    acc.Reload()
    return acc


def _classify_and_extract(
    acc: EventAccumulator,
) -> tuple[dict[str, str], dict[str, list[tuple[int, float]]]]:
    """Walk all tag sources and split into (text_summaries, scalar_series).

    In TF2, ``tf.summary.scalar`` and ``tf.summary.text`` both write to the
    'tensors' plugin. We classify by inspecting the tensor dtype: string
    tensors are text summaries, numeric 0-d tensors are scalars.

    Also merges in any tags from the legacy 'scalars' plugin for safety.
    """
    texts: dict[str, str] = {}
    scalars: dict[str, list[tuple[int, float]]] = {}

    tags = acc.Tags()

    # Legacy scalar plugin (TF1-style)
    for tag in tags.get("scalars", []):
        try:
            events = acc.Scalars(tag)
        except Exception:
            continue
        scalars[tag] = [(e.step, float(e.value)) for e in events]

    # TF2: tensors plugin holds both scalars and text
    for tag in tags.get("tensors", []):
        try:
            events = acc.Tensors(tag)
        except Exception:
            continue
        if not events:
            continue

        # Peek at first event to determine type
        first_tp = events[0].tensor_proto
        is_string = bool(first_tp.string_val) or first_tp.dtype == 7  # DT_STRING=7

        if is_string:
            ev = events[-1]  # latest
            try:
                text = ev.tensor_proto.string_val[0].decode("utf-8", errors="replace")
            except Exception:
                text = str(ev.tensor_proto.string_val[0])
            texts[tag] = text
        else:
            series: list[tuple[int, float]] = []
            for ev in events:
                try:
                    arr = tensor_util.make_ndarray(ev.tensor_proto)
                    val = float(arr.item()) if arr.ndim == 0 else float(arr.flatten()[0])
                except Exception:
                    continue
                series.append((ev.step, val))
            if tag not in scalars:  # don't clobber legacy plugin data
                scalars[tag] = series

    return texts, scalars


def group_by_namespace(
    scalars: dict[str, list[tuple[int, float]]],
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Group scalar tags by their top-level namespace (e.g. 'train', 'eval')."""
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(dict)
    for tag, series in scalars.items():
        if "/" in tag:
            ns, sub = tag.split("/", 1)
        else:
            ns, sub = "(root)", tag
        grouped[ns][sub] = series
    return grouped


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e6 or av < 1e-3:
        return f"{v:.4e}"
    if av >= 100:
        return f"{v:.2f}"
    return f"{v:.4f}"


def render_namespace_table(
    ns: str, tags: dict[str, list[tuple[int, float]]]
) -> str:
    """Render a single unified table for a namespace: rows=steps, cols=metrics."""
    # Build step -> {col: value}
    sub_names = sorted(tags.keys())
    step_rows: dict[int, dict[str, float]] = {}
    for sub in sub_names:
        for step, value in tags[sub]:
            step_rows.setdefault(step, {})[sub] = value

    all_steps = sorted(step_rows.keys())
    lines: list[str] = []

    # Summary stats row (per column)
    stat_lines = ["**Summary:**", "", "| metric | min | max | mean | last | n |", "|---|---|---|---|---|---|"]
    for sub in sub_names:
        values = [v for _, v in tags[sub]]
        if not values:
            stat_lines.append(f"| `{sub}` | – | – | – | – | 0 |")
            continue
        vmin, vmax = min(values), max(values)
        vmean = sum(values) / len(values)
        vlast = values[-1]
        stat_lines.append(
            f"| `{sub}` | {_fmt(vmin)} | {_fmt(vmax)} | {_fmt(vmean)} | {_fmt(vlast)} | {len(values)} |"
        )
    lines.extend(stat_lines)
    lines.append("")

    # Full step-by-step unified table
    lines.append(f"**Full data ({len(all_steps)} steps):**")
    lines.append("")
    header = "| step | " + " | ".join(sub_names) + " |"
    sep = "|------|" + "|".join(["---"] * len(sub_names)) + "|"
    lines.append(header)
    lines.append(sep)
    for step in all_steps:
        row_vals = []
        for sub in sub_names:
            v = step_rows[step].get(sub)
            row_vals.append("" if v is None else _fmt(v))
        lines.append(f"| {step} | " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def render_markdown(trial_path: str, acc: EventAccumulator) -> str:
    texts, scalars = _classify_and_extract(acc)
    grouped = group_by_namespace(scalars)

    trial_name = os.path.basename(os.path.normpath(trial_path))
    out: list[str] = []
    out.append(f"# TensorBoard Trial Report: `{trial_name}`")
    out.append("")
    out.append(f"Source: `{trial_path}`")
    out.append("")

    # --- Section 1: Text summaries ---
    out.append("## 1. Text Summaries")
    out.append("")
    if not texts:
        out.append("_No text summaries found._")
        out.append("")
    else:
        for tag, text in sorted(texts.items()):
            out.append(f"### `{tag}`")
            out.append("")
            out.append(text)
            out.append("")

    # --- Section 2+: One section per scalar namespace ---
    if not grouped:
        out.append("## 2. Scalars")
        out.append("")
        out.append("_No scalars found._")
    else:
        for idx, ns in enumerate(sorted(grouped.keys()), start=2):
            out.append(f"## {idx}. Scalars — `{ns}/`")
            out.append("")
            out.append(render_namespace_table(ns, grouped[ns]))
            out.append("")

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a TensorBoard trial directory to an LLM-friendly markdown report.",
    )
    parser.add_argument(
        "path",
        help="Path to the TensorBoard trial directory (containing events.out.tfevents.* files).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"error: not a directory: {args.path}", file=sys.stderr)
        return 2

    acc = load_accumulator(args.path)
    sys.stdout.write(render_markdown(args.path, acc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
