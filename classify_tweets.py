#!/usr/bin/env python3
import torch
import pandas as pd
import argparse
import json
import os
import re
import time
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from tqdm import tqdm

# --- Hugging Face authentication ---
login("...") # adddd HF login credentials here

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥 Using device: {device}")

# ============================================================
# Utility functions
# ============================================================

def save_checkpoint(results, output_path):
    """Save results to a JSON file with checkpointing"""
    import json
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON
    json_path = output_path.replace('.csv', '.json') if output_path.endswith('.csv') else output_path
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save to CSV format if needed
    if results:
        try:
            # Try to flatten the structure
            flat_results = []
            for batch in results:
                if "output" in batch and isinstance(batch["output"], list):
                    flat_results.extend(batch["output"])

            if flat_results:
                import pandas as pd
                df = pd.DataFrame(flat_results)
                csv_path = json_path.replace('.json', '.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"✅ Also saved CSV: {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")

    print(f"✅ Checkpoint saved: {json_path}")

def load_model_and_tokenizer(model_name):
    """Load model + tokenizer exactly once."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False  # Explicitly set to avoid warnings
    )

    print("✅ Model and tokenizer loaded.")
    return model, tokenizer, generator

def load_prompt_template(template_path):
    """Load the prompt template file."""
    with open(template_path, "r") as f:
        prompt = f.read()

    # Escape braces except {{social_media_posts}}
    prompt = re.sub(r"(?<!\{)\{(?!\{)", "{{", prompt)
    prompt = re.sub(r"(?<!\})\}(?!\})", "}}", prompt)
    return prompt.replace("{{social_media_posts}}", "{social_media_posts}")


def estimate_tokens(tokenizer, text):
    """Token count estimator with margin."""
    return len(tokenizer.encode(text))

def generate_output(prompt, generator, max_new_tokens=1200):
    """Stable text generation with retry backoff."""
    for attempt in range(3):
        try:
            print(f"🤖 Generating output (attempt {attempt+1})...")
            out = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1
            )[0]["generated_text"]

            # Extract only the new text
            result = out[len(prompt):].strip()

            print(f"📝 Generated {len(result)} characters")
            if len(result) < 50:
                print(f"⚠️ Output seems very short: {repr(result)}")

            return result

        except Exception as e:
            print(f"⚠️ Generation failed attempt {attempt+1}: {e}")
            traceback.print_exc()
            time.sleep(5 * (attempt + 1))

    print("❌ All generation attempts failed")
    return None

def extract_json_from_text(text):
    """Extract JSON from text that might contain other content."""
    # Clean the text first
    text = text.strip()

    # Remove common prefixes/suffixes that models add
    # Remove YAML-style separators
    text = re.sub(r'^---+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*---+\s*$', '', text, flags=re.MULTILINE)

    # Remove markdown code blocks (with or without language)
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text)

    # Remove common model prefixes
    prefixes_to_remove = [
        r'^Here(?: is| are|.*?)the (?:results|classifications|output):?\s*',
        r'^Output:?\s*',
        r'^JSON(?: output| response)?:?\s*',
        r'^The (?:classified|extracted) (?:tweets|data):?\s*',
    ]

    for pattern in prefixes_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Try to find JSON array first (most common)
    json_pattern = r'\[\s*\{[\s\S]*?\}\s*\]'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, list) and len(data) > 0:
                # Validate structure
                if all(isinstance(item, dict) for item in data):
                    return data
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                # Fix trailing commas
                fixed = re.sub(r',\s*([\]}])', r'\1', match)
                data = json.loads(fixed)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except:
                continue

    # Try to parse the entire cleaned text
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check if it's a wrapper object
            for key in ['output', 'results', 'classifications', 'data']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If it's a single object, wrap it in a list
            if 'index' in data or 'tweet_id' in data:
                return [data]
    except json.JSONDecodeError:
        pass

    # Last resort: try to find any valid JSON object
    try:
        # Look for individual JSON objects
        obj_pattern = r'\{\s*"[^"]+"\s*:\s*[^}]+?\s*\}'
        obj_matches = re.findall(obj_pattern, text, re.DOTALL)
        if obj_matches:
            objects = []
            for match in obj_matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except:
                    continue
            if objects:
                return objects
    except:
        pass

    return None

def parse_json_or_retry(text, prompt, generator, required_keys):
    """Try to parse JSON repeatedly, correcting errors if needed."""
    for attempt in range(5):
        print(f"\n🔍 Attempt {attempt+1} to parse JSON...")

        # First try to extract JSON from the text
        data = extract_json_from_text(text)

        if data is not None:
            print(f"✅ Extracted {len(data)} items from text")
            # Validate structure
            if isinstance(data, list) and len(data) > 0:
                valid = True
                validated_items = []

                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        print(f"❌ Item {i} is not a dict: {type(item)}")
                        valid = False
                        break

                    # Check for required keys
                    missing_keys = required_keys - set(item.keys())
                    if missing_keys:
                        print(f"❌ Item {i} missing keys: {missing_keys}")
                        print(f"   Item has: {list(item.keys())}")
                        valid = False
                        break

                    # Ensure situational_categories is a list
                    if 'situational_categories' in item:
                        if not isinstance(item['situational_categories'], list):
                            # Try to convert if it's a string
                            if isinstance(item['situational_categories'], str):
                                item['situational_categories'] = [item['situational_categories']]
                            else:
                                print(f"❌ Item {i} situational_categories not a list: {type(item['situational_categories'])}")
                                valid = False
                                break

                    # Ensure rationales is a list
                    if 'rationales' in item:
                        if not isinstance(item['rationales'], list):
                            if isinstance(item['rationales'], str):
                                item['rationales'] = [item['rationales']]
                            else:
                                print(f"❌ Item {i} rationales not a list: {type(item['rationales'])}")
                                valid = False
                                break

                    validated_items.append(item)

                if valid:
                    print(f"✅ Successfully parsed {len(validated_items)} tweets")
                    return validated_items
                else:
                    print(f"❌ Validation failed, retrying...")

        # DEBUG: Show what we're trying to parse
        print(f"📋 Raw text snippet (first 500 chars):")
        print(text[:500])

        # Try to fix common JSON issues before retrying
        if attempt < 2:  # Only try fixing for first few attempts
            print("🛠️ Trying to fix common JSON issues...")
            # Fix trailing commas
            text = re.sub(r',\s*([\]}])', r'\1', text)
            # Fix missing quotes
            text = re.sub(r'([{\[,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
            # Fix unescaped quotes
            text = re.sub(r'(?<!\\)"', r'\"', text)
            continue

        print(f"⚠️ JSON invalid at attempt {attempt+1}. Retrying with new generation...")

        # More specific retry prompt
        retry_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
1. Your response must be EXACTLY and ONLY a valid JSON array.
2. DO NOT include ANY other text, explanations, markdown, or separators.
3. The JSON must start with "[" and end with "]".
4. Each object must have ALL these keys: {list(required_keys)}
5. "situational_categories" must be an ARRAY (list) even if empty.
6. "rationales" must be an ARRAY (list) even if empty.
7. Make sure all strings are properly quoted with double quotes.
8. Do not use single quotes or unquoted strings.

Example of correct format:
[
  {{
    "index": 0,
    "tweet_id": "0",
    "tweet_text": "tweet text here",
    "situational_categories": ["weather_environment"],
    "rationales": ["earthquake"]
  }}
]

Return ONLY this JSON array, nothing else:
"""
        new_text = generate_output(retry_prompt, generator, max_new_tokens=1500)
        if new_text is not None:
            text = new_text
        else:
            print("❌ Failed to generate new text")
            break

    print("❌ Failed to obtain valid JSON after all retries.")
    print("Last raw output was:")
    print(text[:1000] if text else "Empty output")

    # Try one last desperate attempt: extract any valid objects
    print("🔄 Trying desperate extraction of any valid objects...")
    try:
        # Look for any JSON objects
        pattern = r'\{\s*"[^"]*"\s*:\s*[^}]+?\s*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            objects = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except:
                    continue

            if objects:
                print(f"⚠️ Extracted {len(objects)} objects (partial success)")
                return objects
    except:
        pass

    return []

# ============================================================
# Main function
# ============================================================

def main(args):
    model, tokenizer, generator = load_model_and_tokenizer(args.model_name)
    prompt_template = load_prompt_template(args.prompt_template)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.join(args.output_dir, f"{base}_results.json")

    # ============================================================
    # Load existing results FIRST (for resume)
    # ============================================================
    results = []
    processed_indices = set()

    if os.path.exists(output_path):
        print("🟡 Resuming from existing file...")
        with open(output_path) as f:
            results = json.load(f)

        for batch in results:
            for item in batch.get("output", []):
                if "index" in item:
                    processed_indices.add(str(item["index"]))

        print(f"🔁 Found {len(processed_indices)} already-classified tweets")

    # ============================================================
    # Load CSV
    # ============================================================
    print(f"📥 Loading CSV: {args.input_file}")
    df = pd.read_csv(args.input_file)

    df.columns = [str(col).strip() for col in df.columns]

    # ------------------------------------------------------------
    # Column detection helpers
    # ------------------------------------------------------------
    def find_column(target_names, exclude_names=[]):
        for target in target_names:
            for col in df.columns:
                col_lower = col.lower().replace(' ', '_').replace('-', '_')
                if col_lower == target.lower() and col not in exclude_names:
                    return col
        return None

    column_mapping = {}

    # index
    index_col = find_column(['index', 'idx', 'id'])
    if index_col:
        column_mapping['index'] = index_col
    else:
        df['index'] = range(len(df))
        column_mapping['index'] = 'index'

    # tweet_id
    tweet_id_col = find_column(
        ['tweet_id', 'tweetid', 'tweet-id'],
        exclude_names=[column_mapping.get('index')]
    )
    if tweet_id_col:
        column_mapping['tweet_id'] = tweet_id_col
    else:
        df['tweet_id'] = [f"tweet_{i}" for i in range(len(df))]
        column_mapping['tweet_id'] = 'tweet_id'

    # tweet_text
    tweet_text_col = find_column(['tweet_text', 'text', 'tweet', 'content'])
    if tweet_text_col:
        column_mapping['tweet_text'] = tweet_text_col
    else:
        df['tweet_text'] = ""
        column_mapping['tweet_text'] = 'tweet_text'

    # Rename columns
    rename_dict = {
        orig: std for std, orig in column_mapping.items() if orig != std
    }
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Required columns check
    for col in ['index', 'tweet_id', 'tweet_text']:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df['tweet_text'] = df['tweet_text'].fillna('')

    # ============================================================
    # Convert to tweet dicts
    # ============================================================
    tweets = df[['index', 'tweet_id', 'tweet_text']].to_dict('records')

    # Normalize to strings
    for t in tweets:
        t['index'] = str(t['index'])
        t['tweet_id'] = str(t['tweet_id'])

    # ============================================================
    # FILTER already-classified tweets (CORE FIX)
    # ============================================================
    if processed_indices:
        tweets = [
            t for t in tweets
            if t['index'] not in processed_indices
        ]
        print(f"✂️ Skipping already-classified tweets. Remaining: {len(tweets)}")

    if not tweets:
        print("✅ No new tweets to process. Exiting.")
        return

    # ============================================================
    # Batch processing
    # ============================================================
    batch_size = args.batch_size
    SAVE_EVERY = 3

    required_keys = {
        "index",
        "tweet_id",
        "tweet_text",
        "situational_categories",
        "rationales"
    }

    print(f"🔹 Processing {len(tweets)} tweets in batches of {batch_size}")

    for start in tqdm(range(0, len(tweets), batch_size)):
        batch = tweets[start:start + batch_size]

        batch_str = ""
        for tweet in batch:
            batch_str += (
                f"Index: {tweet['index']}, "
                f"Tweet ID: {tweet['tweet_id']}, "
                f"Text: {tweet['tweet_text']}\n"
            )

        prompt = prompt_template.format(social_media_posts=batch_str)

        tokens = estimate_tokens(tokenizer, prompt)
        if tokens > 6000:
            print(f"⚠️ Batch {start} too large ({tokens} tokens), skipping")
            continue

        out_text = generate_output(prompt, generator, max_new_tokens=1000)
        if out_text is None:
            continue

        json_out = parse_json_or_retry(out_text, prompt, generator, required_keys)

        results.append({
            "batch_start": start,
            "batch_end": start + len(batch) - 1,
            "batch_size": len(batch),
            "input_tweets": batch,
            "output": json_out,
            "raw_output": out_text[:1000] + "..." if len(out_text) > 1000 else out_text
        })

        if len(results) % SAVE_EVERY == 0:
            save_checkpoint(results, output_path)

    save_checkpoint(results, output_path)
    print(f"✅ Finished. Results in {output_path}")

    # ============================================================
    # Summary
    # ============================================================
    total_classified = sum(len(b.get("output", [])) for b in results)
    print(f"📊 Total tweets classified: {total_classified}")
    print(f"📊 Total batches stored: {len(results)}")

# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)  # Reduced for better consistency
    args = parser.parse_args()
    main(args)
