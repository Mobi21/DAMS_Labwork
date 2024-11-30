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
    


def call_ollama_with_library_for_keywords(text, model="llama3.1:70b"):
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
    
    
def generate_category_prompt(category_name, keywords, text):
    prompt = (
        f"You are an advanced text analysis model. Analyze the following text and count the occurrences of keywords and phrases related to the category '{category_name}', including duplicates. "
        f"The '{category_name}' category includes terms and concepts such as: {', '.join(keywords)}. "
        "Include synonyms, variations, and any terms associated with these concepts. "
        "When analyzing the text, ignore any weird characters, symbols, or words in other languages. "
        f"Return the count in the format '{category_name}: value'."
        "\n\n"
        "Text to Analyze:\n" + text + "\n\n"
        f"Your only output should be '{category_name}: value. "
        "Do not include any additional text or explanations. REMEMBER THE VALUE SHOULD BE THE COUNT OF KEYWORDS IN THE TEXT."
    )
    return prompt

def count_do_not_track_keywords(text, model="llama3.1"):
    category_name = "do_not_track"
    keywords = ["Do Not Track", "DNT"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'do_not_track: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_data_security_keywords(text, model="llama3.1"):
    category_name = "data_security"
    keywords = ["data security", "security", "secure", "safety", "protect", "data protection", "information security"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'data_security: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_first_party_collection_keywords(text, model="llama3.1"):
    category_name = "first_party_collection"
    keywords = ["first party collection", "collect", "gather", "use", "information"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'first_party_collection: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_third_party_collection_keywords(text, model="llama3.1"):
    category_name = "third_party_collection"
    keywords = ["third party collection", "third party", "third parties", "third-party", "share", "sharing"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'third_party_collection: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_opt_out_keywords(text, model="llama3.1"):
    category_name = "opt_out"
    keywords = ["optout", "opt-out", "opt out"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'opt_out: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_user_choice_keywords(text, model="llama3.1"):
    category_name = "user_choice"
    keywords = ["User Choice", "choice", "control", "revoke", "exercise"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'user_choice: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_data_keywords(text, model="llama3.1"):
    category_name = "data"
    keywords = ["Data", "identifier", "name", "email", "address", "phone number", "ip address", "id", "demographic", "gender", "age", "health", "biometric", "activity", "sleep", "geolocation", "location", "GPS", "photo", "friends", "voice", "video", "inference"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'data: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_legislation_keywords(text, model="llama3.1"):
    category_name = "legislation"
    keywords = ["legislation", "gdpr", "ccpa", "general data protection regulation", "consumer privacy act"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'legislation: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_access_edit_delete_keywords(text, model="llama3.1"):
    category_name = "access_edit_delete"
    keywords = ["Access/Edit/Delete", "access", "edit", "delete", "modify", "revise", "correct", "review", "change", "update"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'access_edit_delete: value'
    except Exception as e:
        print(f"Error: {e}")
        return None

def count_policy_change_keywords(text, model="llama3.1"):
    category_name = "policy_change"
    keywords = ["policy change", "policy modification", "changes", "modifications", "updates", "change", "update"]
    prompt = generate_category_prompt(category_name, keywords, text)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Response is in the format 'policy_change: value'
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
    match = re
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

def parses_response(response, text):
    metrics = {}
    patterns = {
        'do_not_track': r'do_not_track:\s*([0-9]+)',
        'data_security': r'data_security:\s*([0-9]+)',
        'first_party_collection': r'first_party_collection:\s*([0-9]+)',
        'third_party_collection': r'third_party_collection:\s*([0-9]+)',
        'opt_out': r'opt_out:\s*([0-9]+)',
        'user_choice': r'user_choice:\s*([0-9]+)',
        'data': r'data:\s*([0-9]+)',
        'legislation': r'legislation:\s*([0-9]+)',
        'access_edit_delete': r'access_edit_delete:\s*([0-9]+)',
        'policy_change': r'policy_change:\s*([0-9]+)'
    }
    found = False
    match = None
    key = None
    for keys, pattern in patterns.items():
        matches = re.search(pattern, response)
        if matches:
            match = matches
            key = keys
            break  # Exit loop after finding the first match
                
    if match:
        value = match.group(1)
        try:
            metrics[key] = int(value)
        except ValueError:
            metrics[key] = None
            logging.warning(f"Invalid value for {key}: {value}")
    else:
        logging.warning(f"Nothing found in response.")
        metrics[key] = None
    
    return metrics[key]
        

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


def analysis(data):
    results = []
    # List of category functions and their corresponding keys
    category_functions = [
        ('do_not_track', count_do_not_track_keywords),
        ('data_security', count_data_security_keywords),
        ('first_party_collection', count_first_party_collection_keywords),
        ('third_party_collection', count_third_party_collection_keywords),
        ('opt_out', count_opt_out_keywords),
        ('user_choice', count_user_choice_keywords),
        ('data', count_data_keywords),
        ('legislation', count_legislation_keywords),
        ('access_edit_delete', count_access_edit_delete_keywords),
        ('policy_change', count_policy_change_keywords)
    ]
    
    # Wrap data.iterrows() with tqdm for a progress bar
    for _, policy in tqdm(data.iterrows(), total=len(data), desc="Analyzing policies"):
        policy_text = policy["policy_text"]
        
        for category_name, category_function in category_functions:
            found_category = False
            count = 0
            while not found_category:
                count += 1
                print(f"Analyzing {category_name} keywords...")
                found_category = True
                response = category_function(policy_text)
                response = parses_response(response.get("response"), policy_text)
                if response is None:
                    if count > 10:
                        logging.warning(f"Failed to analyze {category_name} keywords.")
                        policy[category_name] = None
                    else:
                        found_category = False
                else:
                    policy[category_name] = response
        
        results.append(policy)
    updated_data = pd.DataFrame(results)
    return updated_data
    
    
def average_results(df):
    for column in df.columns[2:]:
        print(f"Average {column}: {df[column].median()}")
    
def min_results(df):
    for column in df.columns[2:]:
        print(f"Minimum {column}: {df[column].min()}")
        
def max_results(df):
    for column in df.columns[2:]:
        print(f"Maximum {column}: {df[column].max()}")
def null_count(df):
    for column in df.columns[2:]:
        print(f"Null count for {column}: {df[column].isnull().sum()}")  

def combine_df(df1, df2):
    df = pd.concat([df1, df2])
    return df


if __name__ == "__main__":

    df = load_results('final_data.json')
    df = analysis(df)
    save_results(df, 'keyword_results.json')
    
    df1 = load_results('google_play_wayback.json')
    df1 = analysis(df1)
    save_results(df1, 'keyword_wayback.json')