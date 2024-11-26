import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import pickle
def load_data():
    with open("similarity_matrix.json") as file:
        data = json.load(file)
    return data

def save_results(results, filename="results.json"):
    df = pd.DataFrame(results)
    df.to_json(filename, orient="records", indent=4)
    
    
def call_ollama_with_library(text1, text2, model="llama3.1"):
    # Main prompt to guide the model's behavior
    prompt = (
        "You are an advanced text analysis model. Your task is to calculate the semantic similarity "
        "between the two given texts. The similarity score should be a value between 0 (completely dissimilar) "
        "and 1 (identical), based on their meaning, content, and intent.\n"
    )
    
    # Helper instructions to compare the two texts
    prompt_helper1 = (
        "\nFirst Text:\n" + text1 + 
        "\n\nSecond Text:\n" + text2 +
        "\n\nYour only output must be the similarity score in this exact format:\n"
        "Similarity: <value between 0 and 1>\n"
        "No additional explanation or comments are allowed in your response."
    )
    
    try:
        # Use the Ollama library to interact with the model
        response = ollama.generate(
            model=model,
            prompt=prompt + prompt_helper1
        )
        return response  # Return the raw response
    except Exception as e:
        print(f"Error occurred while interacting with the model: {e}")
        return None

    
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
            return None
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None


    
    

from tqdm import tqdm

def compare_policies(data):
    final = []
    count = 0

    # Outer loop progress bar
    outer_loop = tqdm(data.iterrows(), total=len(data), desc="Comparing policies (outer loop)")

    for _, policy in outer_loop:  # Iterate over DataFrame rows
        text1 = policy["policy_text"]
        manufacturer1 = policy["manufacturer"]
        results = []
        count += 1
        count2 = 0

        # Inner loop progress bar
        inner_loop = tqdm(data.iterrows(), total=len(data), desc=f"Policy {count} inner loop", leave=False)
        for _, policy2 in inner_loop:
            if policy2["policy_text"] != text1:
                text2 = policy2["policy_text"]
                response = call_ollama_with_library(text1, text2)
                similarity = parse_response(response)
                if similarity is not None:
                    results.append({
                        "name": policy2["manufacturer"],
                        "similarity": similarity
                    })
                else:
                    results.append({
                        "name": policy2["manufacturer"],
                        "similarity": None
                    })
            count2 += 1
        final.append({
            "name": manufacturer1,
            "series": results
        })
    return final





def load_results(filename="data.json"):
    with open(filename) as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if filename == "results.json":
        df=df[['manufacturer', 'response']]

    return df                
if __name__ == "__main__":
    df = load_results()

    final_df = compare_policies(df)


    save_results(final_df, "similarity_final.json")