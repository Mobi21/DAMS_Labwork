# ambiguity.py
import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import logging
from tqdm import tqdm

def load_results(filename="final_data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_results(results, filename):
    pd.DataFrame(results).to_json(filename, orient="records", indent=4)

def call_ollama_ambiguity(policy_text):
    prompt = """
You are an expert in legal and policy analysis with a specialization in evaluating the clarity and transparency of privacy policies. 
Your task is to analyze the full privacy policy text provided and assign an ambiguity level based on the rubric below.

### Grading Rubric for Privacy Policy Ambiguity
1. NOT AMBIGUOUS (1)
2. SOMEWHAT AMBIGUOUS (2)
3. AMBIGUOUS (3)

Privacy policy text to analyze:
{policy_text}

Your response must be exactly:
Ambiguity_level:<value between 1 and 3>
"""
    try:
        response = ollama.generate(model="llama3.1", prompt=prompt.format(policy_text=policy_text))
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_ambiguity(response_text):
    match = re.search(r"Ambiguity_level:\s*([1-3])", response_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            logging.error("Error parsing ambiguity level.")
            return None
    else:
        logging.error("Error parsing ambiguity level.")
        return None

def analysis(data):
    results = []
    for _, policy in tqdm(data.iterrows(), total=len(data), desc="Ambiguity Analysis"):
        policy_text = policy["policy_text"]
        ambiguity_level = None
        while ambiguity_level is None:
            response_obj = call_ollama_ambiguity(policy_text)
            response_text = response_obj.get("response") if response_obj and hasattr(response_obj, "get") else ""
            ambiguity_level = parse_ambiguity(response_text)
        policy["ambiguity_level"] = ambiguity_level
        results.append(policy)
    return pd.DataFrame(results)

def run_tests(output_dir="results"):
    df1 = load_results("final_data.json")
    df2 = load_results("google_play_wayback.json")
    result1 = analysis(df1)
    result2 = analysis(df2)
    final_results = pd.concat([result1, result2], ignore_index=True)
    save_results(final_results, f"{output_dir}/ambiguity_results.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_tests("results")
