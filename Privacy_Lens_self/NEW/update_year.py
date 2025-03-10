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
    
    **What to look for**:
    - Search for any indication of a last revision or effective date. Example keywords:
      - "Last Updated"
      - "Effective Date"
      - "Revision Date"
      - "Last Revised"
      - "Amended on"
      - "Version Date"
      - "As of"
      - "Posted on"
    - Extract the year from the identified date. If multiple years are mentioned, choose the most recent (i.e., the largest year).
    - If no relevant update year is found, respond with "Last_Updated_Year: 0".
    
    **Required Response Format**:
    Your answer must be exactly one line in either of these two formats:
    - "Last_Updated_Year: YYYY"
    - "Last_Updated_Year: 0"
    
    Nothing else.
    
    Privacy policy text:
        {policy_text}
    
    **Important**:
    - Do not include any additional text, commentary, or explanation in your response.
    - If you find multiple possible dates, pick the largest year.
    - If you are uncertain or no year is explicitly stated, respond with "Last_Updated_Year: 0".
    """
    try:
        response = ollama.generate(model="llama3.1", prompt=prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def parse_year_response(response):
    pattern = r"Last_Updated_Year:\s*(\d{4}|0)"
    match = re.search(pattern, response)
    if match:
        return int(match.group(1))
    else:
        logging.error("Error parsing response.")
        return 0

# Process a single policy for the last update year (with retries and logging)
def process_policy_year(policy):
    updated_policy = policy.copy()
    log_entry = {
        "manufacturer": policy.get("manufacturer", ""),
        "policy_text": policy.get("policy_text", "")
    }
    retries = 0
    found = False
    response_obj = None
    last_year = None
    while not found and retries < 7:
        retries += 1
        response_obj = call_ollama_last_update_year(updated_policy["policy_text"])
        response_text = response_obj.get("response") if response_obj else ""
        parsed_year = parse_year_response(response_text)
        # Accept the parsed year (even if it is 0) as valid
        found = True
        last_year = parsed_year
    updated_policy["last_updated_year"] = last_year
    log_entry["last_update_year_log"] = {
        "response": response_obj,
        "retries": retries
    }
    return updated_policy, log_entry

# Analyze privacy policies for the last update year concurrently
def analyze_last_update_year(data):
    results = []
    logs = []
    records = data.to_dict("records")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for res, log in tqdm(executor.map(process_policy_year, records), total=len(records), desc="Analyzing Policies for Last Update Year"):
            results.append(res)
            logs.append(log)
    return pd.DataFrame(results), pd.DataFrame(logs)

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
