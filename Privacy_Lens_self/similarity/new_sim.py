from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import re
import pickle


# Cache-related functions
CACHE_FILE = "cache.pkl"

def load_cache():
    try:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


# Ollama call with caching
def call_ollama_with_cache(text1, text2, cache, model="llama3.1"):
    pair_key = tuple(sorted((text1, text2)))  # Ensure order-insensitive key
    if pair_key in cache:
        return cache[pair_key]  # Return cached result
    try:
        response = call_ollama_with_library(text1, text2, model)
        cache[pair_key] = response  # Cache the result
        save_cache(cache)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None


# Function to parse Ollama response
def parse_response(response):
    try:
        # Ensure `response` is the expected string, not a dictionary
        if isinstance(response, dict):
            response_text = response.get("response")  # Extract the string part of the response
        else:
            response_text = response

        # Use a regular expression to extract the similarity score
        match = re.search(r"Similarity:\s*([0-9.]+)", response_text)
        if match:
            similarity = float(match.group(1))  # Extract and convert to float
            return similarity
        else:
            print("Similarity score not found in the response.")
            return None
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None


# Policy comparison for a single pair
def compare_policy_pair(policy1, policy2, cache):
    text1 = policy1["policy_text"]
    text2 = policy2["policy_text"]
    response = call_ollama_with_cache(text1, text2, cache)
    similarity = parse_response(response)
    return {
        "name1": policy1["manufacturer"],
        "name2": policy2["manufacturer"],
        "similarity": similarity
    }


# Parallelized comparison of policies
def compare_policies_parallel(data):
    cache = load_cache()  # Load cache
    comparisons = list(combinations(data.to_dict(orient="records"), 2))  # Unique pairwise combinations
    results = []

    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(compare_policy_pair, policy1, policy2, cache)
            for policy1, policy2 in comparisons
        ]
        for task in tasks:
            result = task.result()  # Wait for and collect the result
            if result:
                results.append(result)

    return results


# Main function to run the program
if __name__ == "__main__":
    # Load the dataset
    with open("data.json") as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Perform policy comparisons
    final_results = compare_policies_parallel(df)

    # Save final results
    with open("similarity_final.json", "w") as file:
        json.dump(final_results, file, indent=4)
