import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm

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

# Analyze privacy policies for the last update year
def analyze_last_update_year(data):
    results = []
    for index, policy in tqdm(data.iterrows(), total=len(data), desc="Analyzing Privacy Policies for Last Update Year"):
        policy_text = policy["policy_text"]
        year_found = False
        while not year_found:
            year_found = True
            response = call_ollama_last_update_year(policy_text)
            response = response.get("response")
            #print(response)
            parsed_response = parse_year_response(response)
            print(parsed_response)
            if parsed_response is None:
                year_found = False
            else:
                policy["last_updated_year"] = parsed_response
        results.append(policy)
    return pd.DataFrame(results)

# Generate summary statistics
def summarize_results(data):
    for year in range(2000, 2028):
        count = len(data[data["last_updated_year"] == year])
        print(f"Year: {year}, Count: {count}")
        
    print(len(data))

if __name__ == "__main__":

    
    results = load_results("results.json")
    summarize_results(results)
