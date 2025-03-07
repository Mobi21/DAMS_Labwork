import os
# Set the concurrency level for Ollama: allow 16 parallel requests.
os.environ["OLLAMA_NUM_PARALLEL"] = "16"
# Optionally, you can also adjust the maximum number of loaded models.
# os.environ["OLLAMA_MAX_LOADED_MODELS"] = "4"

import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

# Load JSON data into a DataFrame
def load_results(filename="final_data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Save results (or logs) to JSON
def save_results(results, filename="results.json"):
    pd.DataFrame(results).to_json(filename, orient="records", indent=4)

# Call LLM to compute the semantic similarity between two texts
def call_ollama_similarity(text1, text2, model="llama3.1"):
    prompt = (
        "You are an advanced text analysis model. Your task is to calculate the semantic similarity "
        "between the two given texts. The similarity score should be a value between 0 (completely dissimilar) "
        "and 1 (identical), based on their meaning, content, and intent.\n"
    )
    prompt_helper = (
        "\nFirst Text:\n" + text1 +
        "\n\nSecond Text:\n" + text2 +
        "\n\nYour only output must be the similarity score in this exact format:\n"
        "Similarity: <value between 0 and 1>\n"
        "No additional explanation or comments are allowed in your response."
    )
    try:
        response = ollama.generate(model=model, prompt=prompt + prompt_helper)
        return response  # Expected to be an object with a "response" key
    except Exception as e:
        print(f"Error: {e}")
        return None

# Parse the similarity score from the LLM response
def parse_similarity_response(response_obj):
    # Extract the text using the .get("response") method as in your previous examples.
    response_text = response_obj.get("response") if response_obj and hasattr(response_obj, "get") else ""
    match = re.search(r"Similarity:\s*([0-9.]+)", response_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

# Process a single pair of policies with retry logic and log the raw response
def process_similarity(pair):
    idx1, policy1, idx2, policy2 = pair
    text1 = policy1["policy_text"]
    text2 = policy2["policy_text"]
    retries = 0
    similarity = None
    response_obj = None
    while retries < 7 and similarity is None:
        retries += 1
        response_obj = call_ollama_similarity(text1, text2)
        similarity = parse_similarity_response(response_obj)
    result = {
        "policy1_index": idx1,
        "policy1_manufacturer": policy1["manufacturer"],
        "policy2_index": idx2,
        "policy2_manufacturer": policy2["manufacturer"],
        "similarity": similarity
    }
    log = {
        "policy1_index": idx1,
        "policy2_index": idx2,
        "response": response_obj,
        "retries": retries
    }
    return result, log

# Compare every unique pair of policies concurrently using increased concurrency
def compare_policies(data):
    # Generate all unique pairs (combinations) of policies
    records = list(data.iterrows())  # Each record is (index, row)
    pairs = [(idx1, row1, idx2, row2) 
             for (idx1, row1), (idx2, row2) in combinations(records, 2)]
    
    results = []
    logs = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for res, log in tqdm(executor.map(process_similarity, pairs),
                             total=len(pairs),
                             desc="Comparing policies"):
            results.append(res)
            logs.append(log)
    return results, logs

# Build final structure as a list of dictionaries:
# Each entry contains the policy's manufacturer ("name") and a "series" list of comparisons
def build_final_structure(data, pair_results):
    final_dict = {}
    for idx, row in data.iterrows():
        final_dict[idx] = {"name": row["manufacturer"], "series": []}
    for res in pair_results:
        idx1 = res["policy1_index"]
        idx2 = res["policy2_index"]
        sim = res["similarity"]
        final_dict[idx1]["series"].append({"name": res["policy2_manufacturer"], "similarity": sim})
        final_dict[idx2]["series"].append({"name": res["policy1_manufacturer"], "similarity": sim})
    return list(final_dict.values())

if __name__ == "__main__":
    # Load both datasets and combine them
    df_final = load_results("final_data.json")
    df_wayback = load_results("google_play_wayback.json")
    df_all = pd.concat([df_final, df_wayback], ignore_index=True)
    
    # Compare all policies concurrently (pairwise)
    pair_results, logs = compare_policies(df_all)
    
    # Build the final result structure
    final_structure = build_final_structure(df_all, pair_results)
    
    # Save final results and logs to JSON files
    save_results(final_structure, "similarity_final_results.json")
    save_results(logs, "similarity_final_logs.json")
