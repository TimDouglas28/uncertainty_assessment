#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from pathlib import Path
from transformers import pipeline
import random
import numpy as np
import torch


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MAX_TWEETS_PER_GRID = 30  # deterministic cap to avoid OOM

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_prompt(path):
    return Path(path).read_text()

def format_tweets(df):
    lines = []
    has_uncertainty = "uncertainty_label" in df.columns

    for _, row in df.iterrows():
        tweet_id = row["tweet_id"]
        tweet_text = row["tweet_text"]

        line = f"- {tweet_text} [tweet_id: {tweet_id}]"

        if has_uncertainty and pd.notna(row.get("uncertainty_label")):
            line = f"- {tweet_text} [tweet_id: {tweet_id} - {row['uncertainty_label']}]"

        if "situational_categories" in df.columns and pd.notna(row["situational_categories"]):
            line += f" | categories: {row['situational_categories']}"

        lines.append(line)

    return "\n".join(lines)

def strip_prompt(prompt: str, generated: str) -> str:
    """
    Gemma echoes the prompt. Remove it deterministically.
    """
    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    return generated.strip()

def main(args):
    if args.seed is not None:
        print(f"🔁 Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(args.input_file)

    required_cols = {
        "tweet_id",
        "tweet_text",
        "grid_id",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prompt_template = load_prompt(args.prompt_template)

    # ---- Prompt / data consistency check ----
    prompt_mentions_uncertainty = "uncertainty" in prompt_template.lower()

    if prompt_mentions_uncertainty and "uncertainty_label" not in df.columns:
        raise RuntimeError(
            "Prompt expects uncertainty data, but input CSV has no uncertainty columns."
        )

    if not prompt_mentions_uncertainty and "uncertainty_label" in df.columns:
        print("⚠️  Warning: uncertainty columns present but prompt does not reference them.")

    generator = pipeline(
        "text-generation",
        model=args.model_name,
        device=0
    )

    outputs = []

    # -----------------------------
    # Group by grid cell (explicit order)
    # -----------------------------

    for grid_id in sorted(df["grid_id"].unique()):
        group = df[df["grid_id"] == grid_id]

        # ---- DETERMINISTIC SAMPLING ----
        group_sorted = group.sort_values(by=["tweet_id"], ascending=True)
        sampled_group = group_sorted.head(MAX_TWEETS_PER_GRID)

        tweets_block = format_tweets(sampled_group)

        prompt = prompt_template
        prompt = prompt.replace("{{event}}", str(args.event))
        prompt = prompt.replace("{{grid_id}}", str(grid_id))
        prompt = prompt.replace("{{num_tweets}}", str(len(sampled_group)))
        prompt = prompt.replace("{{tweets}}", tweets_block)

        # ---- HARD SAFETY CHECK ----
        if "{tweets}" in prompt or "{grid_id}" in prompt:
            raise RuntimeError(
                "❌ Prompt formatting failed — unresolved template variables remain."
            )

        raw = generator(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_full_text=True
        )[0]["generated_text"]

        report_text = strip_prompt(prompt, raw)

        # ---- SECOND SAFETY CHECK ----
        if report_text.startswith("You are an AI system"):
            raise RuntimeError(
                "❌ Model echoed prompt. Output is invalid."
            )

        outputs.append({
            "event": args.event,
            "grid_id": int(grid_id),
            "num_tweets": len(sampled_group),
            "max_tweets_per_grid": MAX_TWEETS_PER_GRID,
            "tweet_ids": sampled_group["tweet_id"].tolist(),
            "report_text": report_text
        })

    # -----------------------------
    # Save output
    # -----------------------------
    out_path = Path(args.output_dir) / "reports.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"✅ Reports written to {out_path}")

# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--event", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--prompt_template", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    main(args)
