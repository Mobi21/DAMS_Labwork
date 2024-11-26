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


def load_results(filename="data.json"):
    with open(filename, encoding='utf-8') as file:

        data = json.load(file)
    df = pd.DataFrame(data)
    if filename == "results.json":
        df=df[['manufacturer', 'response']]

    return df 

def save_results(results, filename="results.json"):
    df = pd.DataFrame(results)
    df.to_json(filename, orient="records", indent=4)
    


def call_ollama_with_library_for_keywords(text, model="llama3.1"):
    # Define the prompt for analyzing the text and counting keywords
    prompt = (
        "You are an advanced text analysis model. Analyze the following text and count the occurrences of keywords including duplicates "
        "for each category listed below. Return the counts in a structured JSON format where each category is a key, "
        "and the count of its corresponding keywords is the value. Including Duplicates. The keywords don't have to match the category name but just generally have to be in the category. The selection of what words matches the categories can be loose. Find as many as you can for each category including duplicates. \n\n"
        "When analyzing the text Ignore any weird characters, symbols, or words in other languages. Completely ignore when analyzing the text"
        "Categories:\n"
        "{\n"
        '    "do_not_track"\n'
        '    "data_security",\n'
        '    "first_party_collection": ,\n'
        '    "third_party_collection":,\n'
        '    "opt_out":,\n'
        '    "user_choice":\n'
        '    "data":,\n'
        '    "legislation":,\n'
        '    "access_edit_delete":,\n'
        '    "policy_change": \n'
        "}\n\n"
        "Text to Analyze:\n" + text + "\n\n"

        "Your only output should be the count of each category. MAKE SURE THE CATEGORY NAMES are do_not_track, data_security, first_party_collection, third_party_collection, opt_out, user_choice, data, legislation, access_edit_delete, policy_change. THERE should be no other categories:\n"
        "\n"
        '    "do_not_track": (value),\n'
        '    "data_security": (value),\n'
        '    "first_party_collection": (value),\n'
        '    "third_party_collection": (value),\n'
        '    "opt_out": (value),\n'
        '    "user_choice": (value),\n'
        '    "data": (value),\n'
        '    "legislation": (value),\n'
        '    "access_edit_delete": (value),\n'
        '    "policy_change": (value)\n'
        "\n"
        "Remember to provide no other output ot text to your response.The category names should be exactly the same as above including the underscore.This should all happen in one response. Do not provide multiple responses or any additional information."
    )
    
    try:
        # Use Ollama library to interact with the model
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response should contain the JSON keyword counts
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_response(response):
    """
    Parses the keyword-counting response and extracts numerical metrics using regex patterns.
    Ensures the metrics are returned in the specified order.
    
    Args:
        response (str): The response as a string containing metric information.
    
    Returns:
        dict: A dictionary with metric names as keys and their extracted values as values, 
              ordered as per the given categories.
    """
    # Categories to extract in the desired order
    categories = [
        'do_not_track', 
        'data_security', 
        'first_party_collection', 
        'third_party_collection', 
        'opt_out', 
        'user_choice', 
        'data', 
        'legislation', 
        'access_edit_delete', 
        'policy_change'
    ]
    # Extract response text if it is a dictionary
    if isinstance(response, dict):
        response_text = response.get("response", "")
    else:
        response_text = response
    
    # Regex to extract key-value pairs
    pattern = r'"([\w_]+)":\s*([0-9]+)'
    matches = re.findall(pattern, response_text)
    metrics = {}
    nigga = False
    if len(matches) != len(categories):
        logging.warning(f"Expected {len(categories)} metrics, but found {len(matches)}.")
        
    # Convert matches into a dictionary
    extracted_metrics = {key: int(value) for key, value in matches}

    # Populate the final metrics dictionary in the desired order
    if nigga == False:
        for category in categories:
            if category in extracted_metrics:
                metrics[category] = extracted_metrics[category]
            else:
                metrics[category] = None  # Default value for missing categories
                logging.warning(f"Metric '{category}' not found in response. Defaulting to None.")
    
    return metrics

        

def analyze_keywords(data):
    results = []
    main_loop = tqdm(data.iterrows(), total=len(data), desc="Keyword Analysis")
    for _, policy in main_loop:
        valid = False
        runcount = 0
        while(valid == False):
            runcount += 1
            valid = True
            text = policy["policy_text"]
            response = call_ollama_with_library_for_keywords(text)

            metrics = parse_response(response)
            if runcount > 14:
                runcount = 0
                results.append({"manufacturer": policy["manufacturer"], **metrics})
            elif None in metrics.values():
                valid = False
            elif None not in metrics.values():
                result = {"manufacturer": policy["manufacturer"], **metrics}
                runcount = 0
                results.append(result)

    return results
   
def average_results(df):
    for column in df.columns[1:]:
        print(f"Average {column}: {df[column].median()}")
    
def min_results(df):
    for column in df.columns[1:]:
        print(f"Minimum {column}: {df[column].min()}")
        
def max_results(df):
    for column in df.columns[1:]:
        print(f"Maximum {column}: {df[column].max()}")
def null_count(df):
    for column in df.columns[1:]:
        print(f"Null count for {column}: {df[column].isnull().sum()}")  

def combine_df(df1, df2):
    df = pd.concat([df1, df2])
    return df


if __name__ == "__main__":

    df = load_results('Wayback_true.json')
    df1 = load_results('keyword_analysis_data.json')
    
    combined_df = combine_df(df, df1)
    
    average_results(combined_df)
    min_results(combined_df)
    max_results(combined_df)
    
    
