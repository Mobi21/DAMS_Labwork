import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm
import random

def load_results(filename="data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    return df


def save_results(results, filename="results.json"):
    results.to_json(filename, orient="records", indent=4)
    
    
def call_ollama(policy_text):
    prompt = """
        You are an expert in legal text analysis, specializing in privacy policies. Your task is to analyze the following privacy policy text and determine its level of ambiguity based on the criteria below.

        **Ambiguity Levels:**

        1. **Not Ambiguous (1):**


        2. **Somewhat Ambiguous (2):**


        3. **Ambiguous (3):**


        **Instructions:**

        - Carefully read the provided privacy policy text.
        - Determine the level of ambiguity based on the criteria above.
        - Do not include any additional commentary or analysis.
        - **Your response should strictly be in the following format:**

        Ambiguity_level:[value]


        The privacy policy text to analyze is provided below:
            {policy_text}
            \n
            
        Your response should only be:
        "Ambiguity_level":value(1-3)
        
        STRICT OUTPUT REQUIREMENT: Do not include any additional information, text, or explanation. 
        Your response must be in the format of Ambiguity_level:value(1-3). THERE SHOULD BE NO OTHER NUMBERS IN THE RESPONSE
    """
    try:
        response = ollama.generate(model="llama3.1", prompt= prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    
    
def parse_response(response):
    pattern = r'"Ambiguity_level":\s*([1-3])'
    match = re.search(r"1|2|3", response)
    
    if match:
        try:
            return float(match.group())
        except ValueError:
            logging.error("Error parsing response.")
            return None  
    else:
        logging.error("Error parsing response.")
        return None

def analysis(data):
    results = []
    for index, policy in tqdm(data.iterrows(), total=len(data), desc="Analyzing Privacy Policies"):
        policy_text = policy["policy_text"]
        
        found_ambiguity = False
        while not found_ambiguity:
            found_ambiguity = True
            response = call_ollama(policy_text)
            response = parse_response(response.get("response"))
            if response is None:
                found_ambiguity = False
            else:
                policy["ambiguity_level"] = response
        
        results.append(policy)
            
    return pd.DataFrame(results)

def results(data):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for index, policy in data.iterrows():
        if policy["ambiguity_level"] == 1:
            count_1 += 1
        elif policy["ambiguity_level"] == 2:
            count_2 += 1
        elif policy["ambiguity_level"] == 3:
            count_3 += 1
            
    print(f"Total Policies: {len(data)}")
    print(f"Ambiguity Level 1: {count_1}")
    print(f"Ambiguity Level 2: {count_2}")
    print(f"Ambiguity Level 3: {count_3}")
   
if __name__ == "__main__":
    
    # Load the data
    data = load_results("final_data.json")

    """
    # Analyze the privacy policies
    results = analysis(data)
    
    # Save the results
    save_results(results, "results.json")
    
    data = load_results("google_play_wayback.json")
    results = analysis(data)
    save_results(results, "wayback_results.json")
    """
    # Load the results
    result = load_results("wayback_results.json")
    results(result)