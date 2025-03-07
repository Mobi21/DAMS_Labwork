# update_year.py
import json
import re
import pandas as pd
import ollama
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_results(filename="final_data.json"):
    with open(filename, encoding='utf-8') as file:
         data = json.load(file)
    return pd.DataFrame(data)

def save_results(results, filename):
    pd.DataFrame(results).to_json(filename, orient="records", indent=4)

def call_ollama_last_update_year(policy_text):
    prompt = f"""
You are an advanced policy analysis AI. Your task is to examine the following privacy policy text and find the year of its most recent update.
Search for keywords such as "Last Updated", "Effective Date", "Revision Date", etc.
If found, respond with: "Last_Updated_Year: YYYY"
If not found or uncertain, respond with: "Last_Updated_Year: 0"

Policy text:
{policy_text}
"""
    try:
        response = ollama.generate(model="llama3.1", prompt=prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_year_response(response_text):
    pattern = r"Last_Updated_Year:\s*(\d{4}|0)"
    match = re.search(pattern, response_text)
    if match:
        return int(match.group(1))
    else:
        logging.error("Error parsing update year.")
        return 0

def process_policy_year(policy):
    updated_policy = policy.copy()
    log_entry = {"manufacturer": policy.get("manufacturer", ""), "policy_text": policy.get("policy_text", "")}
    retries = 0
    last_year = None
    while retries < 7:
        retries += 1
        response_obj = call_ollama_last_update_year(updated_policy["policy_text"])
        response_text = response_obj.get("response") if response_obj and hasattr(response_obj, "get") else ""
        last_year = parse_year_response(response_text)
        break  # Accept result even if 0
    updated_policy["last_updated_year"] = last_year
    log_entry["last_update_year_log"] = {"response": response_obj, "retries": retries}
    return updated_policy, log_entry

def analyze_last_update_year(data):
    results = []
    logs = []
    records = data.to_dict("records")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for res, log in tqdm(executor.map(process_policy_year, records), total=len(records), desc="Update Year Analysis"):
            results.append(res)
            logs.append(log)
    return results, logs

def run_tests(output_dir="results"):
    df1 = load_results("final_data.json")
    results1, logs1 = analyze_last_update_year(df1)
    df2 = load_results("google_play_wayback.json")
    results2, logs2 = analyze_last_update_year(df2)
    final_results = results1 + logs1  # Actually we want results separately...
    final_results = results1 + results2
    final_logs = logs1 + logs2
    save_results(final_results, f"{output_dir}/last_update_final_results.json")
    save_results(final_logs, f"{output_dir}/last_update_final_logs.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_tests("results")
