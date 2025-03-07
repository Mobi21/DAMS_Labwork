import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load JSON data
def load_results(filename="final_data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    return df

# Save results to JSON
def save_results(results, filename="results.json"):
    results.to_json(filename, orient="records", indent=4)

# Call LLM to find the last update year
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

# Parse the LLM response
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

# Generate summary statistics
def summarize_results(data):
    for year in range(2000, 2028):
        count = len(data[data["last_updated_year"] == year])
        print(f"Year: {year}, Count: {count}")
    print(f"Total policies: {len(data)}")

if __name__ == "__main__":
    # Load data from both sources
    data_final = load_results("final_data.json")
    data_wayback = load_results("google_play_wayback.json")
    
    # Analyze policies concurrently
    results_final, logs_final = analyze_last_update_year(data_final)
    results_wayback, logs_wayback = analyze_last_update_year(data_wayback)
    
    # Combine both datasets so that final results and logs have the same number of rows
    final_results = pd.concat([results_final, results_wayback], ignore_index=True)
    final_logs = pd.concat([logs_final, logs_wayback], ignore_index=True)
    
    # Save the combined final results and logs
    save_results(final_results, "last_update_final_results.json")
    save_results(final_logs, "last_update_final_logs.json")
    
    # Optionally, generate summary statistics
    summarize_results(final_results)
