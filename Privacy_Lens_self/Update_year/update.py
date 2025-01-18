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
    prompt = """
    You are a legal and policy analysis expert. Your task is to analyze the privacy policy provided and extract the year of the most recent update.

    **Instructions:**
    - Carefully examine the policy text for any mention of "Last Updated," "Effective Date," or similar phrases.
    - Extract the year from the date mentioned. For example:
        - If the text states "Last Updated: May 1, 2022," you should extract "2022."
        - If multiple years are mentioned, use the most recent year.
    - If no year is mentioned, respond with "Last_Updated_Year: 0".

    **Response Format:**
    Your response should strictly be in one of the following formats:
        - "Last_Updated_Year: [YYYY]"
        - "Last_Updated_Year: 0"

    Privacy policy text to analyze:
    {policy_text}

    **Important Note:** Do not include any additional text or commentary in your response. Strictly adhere to the response format.
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
            parsed_response = parse_year_response(response.get("response"))
            if parsed_response is None:
                year_found = False
            else:
                policy["last_updated_year"] = parsed_response
        results.append(policy)
    return pd.DataFrame(results)

# Generate summary statistics
def summarize_results(data):
    policies_with_year = len(data[data["last_updated_year"] > 0])
    policies_without_year = len(data[data["last_updated_year"] == 0])
    
    print(f"Total Policies: {len(data)}")
    print(f"Policies with 'Last Updated Year': {policies_with_year}")
    print(f"Policies without 'Last Updated Year': {policies_without_year}")

if __name__ == "__main__":
    # Load the first dataset
    data = load_results("final_data.json")

    # Analyze the last update year
    results = analyze_last_update_year(data)

    # Save the results
    save_results(results, "results.json")

    # Load the second dataset
    data = load_results("google_play_wayback.json")

    # Analyze the last update year
    results = analyze_last_update_year(data)

    # Save the results
    save_results(results, "wayback_results.json")
