# keyword.py
import json
import pandas as pd
import re
import ollama
import logging
from tqdm import tqdm

def load_results(filename="final_data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_results(results, filename):
    pd.DataFrame(results).to_json(filename, orient="records", indent=4)

def call_ollama_keywords(text, category, description, model="llama3.1"):
    prompt = f"""
You are a highly skilled text analysis assistant specializing in keyword extraction.
Identify all occurrences of keywords related to the category: "{category}"
Description: "{description}"
Analyze the following text:
{text}
Your output must be a single JSON array of keywords, e.g., ["keyword1", "keyword2"].
If no keywords are found, output an empty array: [].
"""
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_keywords(response_text):
    try:
        matches = re.findall(r'"(.*?)"', response_text)
        return len(matches)
    except Exception as e:
        logging.error(f"Error parsing keywords: {e}")
        return 0

def process_policy_keywords(policy, categories):
    result = {"manufacturer": policy["manufacturer"]}
    logs = []
    policy_text = policy["policy_text"]
    for category, description in categories.items():
        retries = 0
        valid = False
        response_obj = None
        keyword_count = 0
        while not valid and retries < 5:
            retries += 1
            response_obj = call_ollama_keywords(policy_text, category, description)
            response_text = response_obj.get("response") if response_obj and hasattr(response_obj, "get") else ""
            keyword_count = parse_keywords(response_text)
            valid = True  # Accept result even if count is zero
        result[category] = keyword_count
        log_entry = {
            "manufacturer": policy["manufacturer"],
            "policy_text": policy_text,
            "category": category,
            "retries": retries
        }
        if response_obj:
            log_entry.update(response_obj)
        logs.append(log_entry)
    return result, logs

def collect_keyword_data(data, categories):
    results = []
    all_logs = []
    for policy in tqdm(data.to_dict("records"), total=len(data), desc="Keyword Collection"):
        res, log = process_policy_keywords(policy, categories)
        results.append(res)
        all_logs.extend(log)
    return results, all_logs

def run_tests(output_dir="results"):
    categories = {
        "do_not_track": "Keywords related to 'Do Not Track' functionality or user requests to minimize tracking.",
        "data_security": "Keywords related to data protection, encryption, and security measures.",
        "first_party_collection": "Keywords about data collected directly by the company.",
        "third_party_collection": "Keywords about data sharing or collection by third parties.",
        "opt_out": "Keywords about users opting out of data collection or specific services.",
        "user_choice": "Keywords about user preferences, control, or decision-making regarding data.",
        "data": "Generic terms related to data or information, including personal and demographic data.",
        "legislation": "Keywords referencing privacy laws, regulations, or compliance standards.",
        "access_edit_delete": "Keywords about accessing, editing, or deleting personal data.",
        "policy_change": "Keywords about modifications, updates, or changes to privacy policies."
    }
    df1 = load_results("final_data.json")
    results1, logs1 = collect_keyword_data(df1, categories)
    df2 = load_results("google_play_wayback.json")
    results2, logs2 = collect_keyword_data(df2, categories)
    final_results = results1 + results2
    final_logs = logs1 + logs2
    save_results(final_results, f"{output_dir}/keyword_results.json")
    save_results(final_logs, f"{output_dir}/keyword_logs.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_tests("results")
