#!/usr/bin/env python3
import torch
import pandas as pd
import argparse
import json
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm

# -------------------- PAGER --------------------
from pager import (
    PAGER_TABLE_NAPA_2014,
    PAGER_TABLE_CHILE_2014,
    PAGER_TABLE_NEPAL_2015,
    PAGER_TABLE_RIDGECREST_2019,
    PAGER_TABLE_FUKUSHIMA_2021,
    PAGER_TABLE_HAITI_2021,
    PAGER_TABLE_TESTVILLE_SYNTHETIC_2015,
)

# -------------------- AUTH --------------------
login("...") # Add HF token here

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥 Using device: {device}")

# =====================================================================
# ---------------- MMI FLOAT → ROMAN → PAGER SUMMARY ------------------
# =====================================================================

def mmi_float_to_intensity_band(mmi_float):
    val = int(round(float(mmi_float)))
    val = max(1, min(val, 10))
    mapping = {
        1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
        6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"
    }
    return mapping[val]

def build_pager_summary(mmi_roman, table):
    row = table.get(mmi_roman)
    if row is None:
        return f"No PAGER data available for intensity {mmi_roman}."
    return (
        f"MMI {mmi_roman}:\n"
        f"- Perceived shaking: {row['perceived_shaking']}\n"
        f"- Potential damage (resistant structures): {row['potential_damage_resistant']}\n"
        f"- Potential damage (vulnerable structures): {row['potential_damage_vulnerable']}\n"
        f"- Estimated population exposed: {row['population_exposed']}"
    )

# =====================================================================
# ----------------------- MODEL LOADING -------------------------------
# =====================================================================

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

# =====================================================================
# ----------------------- GENERATION ----------------------------------
# =====================================================================

@torch.inference_mode()
def generate_output(model, tokenizer, prompt, temperature, max_new_tokens=600):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    for attempt in range(3):
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        except Exception as e:
            print(f"⚠️ Generation failed (T={temperature}, attempt {attempt + 1}): {e}")
            time.sleep(2)
    return None

def generate_M_samples(model, tokenizer, prompt, temperatures):
    outputs = []
    for T in temperatures:
        out = generate_output(model, tokenizer, prompt, temperature=T)
        if out:
            outputs.append({"temperature": T, "output": out})
    return outputs

# =====================================================================
# ---------------------------- MAIN -----------------------------------
# =====================================================================

def main(args):

    # ---------------- Load context ----------------
    with open(args.crisis_context_file) as f:
        crisis_context = json.load(f)

    matches = [k for k in crisis_context if args.event.lower() in k.lower()]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve crisis for event='{args.event}'")
    crisis_key = matches[0]
    crisis_info = crisis_context[crisis_key]
    print(f"📍 Using crisis context: {crisis_key}")

    # ---------------- PAGER table ----------------
    PAGER_TABLES = {
        "napa": PAGER_TABLE_NAPA_2014,
        "chile": PAGER_TABLE_CHILE_2014,
        "nepal": PAGER_TABLE_NEPAL_2015,
        "ridgecrest_2019": PAGER_TABLE_RIDGECREST_2019,
        "fukushima": PAGER_TABLE_FUKUSHIMA_2021,
        "haiti": PAGER_TABLE_HAITI_2021,
        "testville_synthetic_2015": PAGER_TABLE_TESTVILLE_SYNTHETIC_2015,
        "ridgecrest_synthetic": PAGER_TABLE_RIDGECREST_2019
    }
    event_key = args.event.lower()
    pager_table = PAGER_TABLES[event_key]

    # ---------------- Model ----------------
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # ---------------- Output ----------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "calibration_results.json")

    # ---------------- Load existing results (RESUME) ----------------
    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
        calibrated_indices = {r["index"] for r in results}
        print(f"🔁 Resuming calibration — {len(calibrated_indices)} tweets already processed")
    else:
        results = []
        calibrated_indices = set()
        print("🆕 Starting fresh calibration run")

    # ---------------- Data ----------------
    df = pd.read_csv(args.input_file)
    with open(args.prompt_template) as f:
        template = f.read()

    # ---------------- Loop ----------------
    for _, row in tqdm(df.iterrows(), total=len(df)):

        if pd.isna(row["MMI"]):
            continue

        if int(row["index"]) in calibrated_indices:
            continue

        mmi_roman = mmi_float_to_intensity_band(row["MMI"])
        pager_summary = build_pager_summary(mmi_roman, pager_table)

        social_post_block = (
            f"index: {row['index']}\n"
            f"tweet_id: {row['tweet_id']}\n"
            f"tweet_text: {row['tweet_text']}\n"
            f"situational_categories: {row['situational_categories']}\n"
            f"rationales: {row['rationales']}\n"
        )

        prompt = (
            template
            .replace("{{EVENT_NAME}}", crisis_info["crisis_name"])
            .replace("{{REGION}}", crisis_key)
            .replace("{{TIME}}", str(row["time"]))
            .replace("{{PAGER_SUMMARY}}", pager_summary)
            .replace("{{SOCIAL_MEDIA_POSTS}}", social_post_block)
        )

        samples = generate_M_samples(model, tokenizer, prompt, [0.5, 1.0, 1.5])
        if not samples:
            continue

        results.append({
            "index": int(row["index"]),
            "tweet_id": row["tweet_id"],
            "time": row["time"],
            "mmi_roman": mmi_roman,
            "situational_categories": row["situational_categories"],
            "llm_outputs_M": samples
        })

        # Incremental write
        tmp = out_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp, out_file)

    print(f"✅ Calibration finished. Total calibrated tweets: {len(results)}")

# =====================================================================
# ----------------------------- CLI -----------------------------------
# =====================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--prompt_template", required=True)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--crisis_context_file", required=True)
    args = p.parse_args()
    main(args)
