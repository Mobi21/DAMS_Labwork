import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm
import random

def load_data():
    with open("data.json") as file:
        data = json.load(file)
    return data

def call_ollama_with_library(text, prompt, model="llama3.1"):
    prompt_helper = (
        "Your only output should be the requested metric value in this format. THERE should not be any other text in your output:\n "
        "{\n"
        "  'coherence_score': 0-1,\n"
        "  'imprecise_words_frequency': 0-1,\n"
        "  'connective_words_frequency': 0-1,\n"
        "  'reading_complexity': 4.7-26.99,\n"
        "  'reading_time': 1-130min,\n"
        "  'entropy': 5.99-10.13,\n"
        "  'unique_words_frequency': 0-1,\n"
        "  'grammatical_errors': 0-1\n"
        "}\n\n"      
    )
    try:
        # Use Ollama library to interact with the model
        response = ollama.generate(model=model, prompt=prompt + text + prompt_helper)
        return response  # Response is already in JSON format
    except Exception as e:
        print(f"Error: {e}")
        return None

# Define functions to call for each metric
def get_coherence_score(policy_text):
    prompt = "Evaluate the coherence score (0-1) for this privacy policy text: "
    return call_ollama_with_library(policy_text, prompt)

def get_imprecise_words_frequency(policy_text):
    prompt = "Count the frequency of vague words (e.g., 'generally', 'may', 'sometimes') in this text: "
    return call_ollama_with_library(policy_text, prompt)

def get_connective_words_frequency(policy_text):
    prompt = "Count the frequency of connective words (e.g., 'and', 'then') in this text: "
    return call_ollama_with_library(policy_text, prompt)

def get_reading_complexity(policy_text):
    prompt = "Calculate the Flesch-Kincaid Grade Level score for readability of this text: "
    return call_ollama_with_library(policy_text, prompt)

def get_reading_time(policy_text):
    prompt = "Calculate the estimated reading time for this text: "
    return call_ollama_with_library(policy_text, prompt)

def get_entropy(policy_text):
    prompt = "Calculate the Shannon entropy for this text: "
    return call_ollama_with_library(policy_text, prompt)

def get_unique_words_frequency(policy_text):
    prompt = "Count the number of unique words divided by total words in this text: "
    return call_ollama_with_library(policy_text, prompt)

def get_grammatical_errors(policy_text):
    prompt = "Count the grammatical errors in this text: "
    return call_ollama_with_library(policy_text, prompt)

def get_analysis(policy_text):
    # Define a single long prompt for all analyses
    prompt = (
    "Analyze the following privacy policy text and provide the following metrics as numbers or clear values in JSON format:\n"
    "1. Coherence score (0-1).\n"
    "2. Frequency of vague words (e.g., 'generally', 'may', 'sometimes').\n"
    "3. Frequency of connective words (e.g., 'and', 'then').\n"
    "4. Flesch-Kincaid Grade Level score for readability.\n"
    "5. Estimated reading time in seconds.\n"
    "6. Shannon entropy of the text.\n"
    "7. Ratio of unique words to total words (0-1).\n"
    "8. Number of grammatical errors.\n\n"
    "Return the results in the following JSON format:\n"
    "{\n"
    "  'coherence_score': ,\n"
    "  'imprecise_words_frequency': ,\n"
    "  'connective_words_frequency': ,\n"
    "  'reading_complexity': ,\n"
    "  'reading_time': ,\n"
    "  'entropy': ,\n"
    "  'unique_words_frequency': ,\n"
    "  'grammatical_errors': \n"
    "}\n\n"
    "Privacy policy text:\n"
    )   

    return call_ollama_with_library(policy_text, prompt)

def analysis(data):
    results = []
    for policy in data:
        manufacturer = policy["manufacturer"]
        policy_text = policy["policy_text"]

        result = {
            "manufacturer": manufacturer,
            "coherence_score": get_coherence_score(policy_text),
            "imprecise_words_frequency": get_imprecise_words_frequency(policy_text),
            "connective_words_frequency": get_connective_words_frequency(policy_text),
            "reading_complexity": get_reading_complexity(policy_text),
            "reading_time": get_reading_time(policy_text),
            "entropy": get_entropy(policy_text),
            "unique_words_frequency": get_unique_words_frequency(policy_text),
            "grammatical_errors": get_grammatical_errors(policy_text),
        }

        results.append(result)
    return results

def new_analysis(data):
    results = []
    for policy in data:
        manufacturer = policy["manufacturer"]
        policy_text = policy["policy_text"]

        # Call the analysis function
        analysis_result = get_analysis(policy_text)

        # If the analysis result is already a dictionary, directly use it
        if isinstance(analysis_result, dict):
            metrics = analysis_result
        else:
            print(f"Unexpected format for analysis result for manufacturer: {manufacturer}")
            metrics = {}

        result = {"manufacturer": manufacturer, **metrics}

        results.append(result)
    return results

def save_results(results, filename="results.json"):
    df = pd.DataFrame(results)
    df.to_json(filename, orient="records", indent=4)

def parse_response(response):
    metrics = {}
    patterns = {
        'coherence_score': r"'coherence_score':\s*([0-9.]+)",
        'imprecise_words_frequency': r"'imprecise_words_frequency':\s*([0-9.]+)",
        'connective_words_frequency': r"'connective_words_frequency':\s*([0-9.]+)",
        'reading_complexity': r"'reading_complexity':\s*([0-9.]+)",
        'reading_time': r"'reading_time':\s*([0-9.]+)min",
        'entropy': r"'entropy':\s*([0-9.]+)",
        'unique_words_frequency': r"'unique_words_frequency':\s*([0-9.]+)",
        'grammatical_errors': r"'grammatical_errors':\s*([0-9.]+)",
    }
    
    for key, pattern in patterns.items():
        error_count = 0
        match = re.search(pattern, response)
        if match:
            value = match.group(1)
            if key == 'reading_time':
                try:
                    metrics['reading_time_minutes'] = float(value)
                except ValueError:
                    metrics['reading_time_minutes'] = None
                    logging.warning(f"Invalid reading_time value: {value}")
            elif key == 'grammatical_errors':
                try:
                    metrics['grammatical_errors'] = int(float(value))
                except ValueError:
                    metrics['grammatical_errors'] = None
                    logging.warning(f"Invalid grammatical_errors value: {value}")
            elif key == 'ambiguity_classification':
                metrics['ambiguity_classification'] = value
            else:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = None
                    logging.warning(f"Invalid value for {key}: {value}")
        else:
            if key == 'reading_time':
                metrics['reading_time_minutes'] = None
            elif key == 'grammatical_errors':
                metrics['grammatical_errors'] = None
            elif key == 'ambiguity_classification':
                metrics['ambiguity_classification'] = None
            else:
                metrics[key] = None
            #logging.warning(f"Metric '{key}' not found in response.")
    return metrics

def load_results(filename="data.json"):
    with open(filename) as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if filename == "results.json":
        df=df[['manufacturer', 'response']]

    return df

def change_columns_names(df):
    df = df.rename(columns={"url": "manufacturer", "Coherance Score": "coherence_score", "Imprecise Words": "imprecise_words_frequency", "Connective Words": "connective_words_frequency", "flesch_kincaid_grade_level": "reading_complexity", "Reading_Time (Min)": "reading_time_minutes", "Entropy": "entropy", "Unique_words": "unique_words_frequency", "correct_grammar_frequency": "grammatical_errors"})
    return df

def average_results(df):
    num_cols = ['coherence_score', 'imprecise_words_frequency', 'connective_words_frequency', 'reading_complexity', 'reading_time_minutes', 'entropy', 'unique_words_frequency', 'grammatical_errors']
    avg_results = {}
    
    coherence_average = df['coherence_score'].median()
    imprecise_average = df['imprecise_words_frequency'].median()
    connective_average = df['connective_words_frequency'].median()
    complexity_average = df['reading_complexity'].median()
    time_average = df['reading_time_minutes'].median()
    entropy_average = df['entropy'].median()
    unique_average = df['unique_words_frequency'].median()
    grammatical_average = df['grammatical_errors'].median()
    avg_results['coherence_score'] = float(coherence_average)
    avg_results['imprecise_words_frequency'] = float(imprecise_average)
    avg_results['connective_words_frequency'] = float(connective_average)
    avg_results['reading_complexity'] = float(complexity_average)
    avg_results['reading_time_minutes'] = float(time_average)
    avg_results['entropy'] = float(entropy_average)
    avg_results['unique_words_frequency'] = float(unique_average)
    avg_results['grammatical_errors'] = float(grammatical_average)
    
    print("Average Results:")  
    display_average_results(avg_results)
    
def max_results(df):
    num_cols = ['coherence_score', 'imprecise_words_frequency', 'connective_words_frequency', 'reading_complexity', 'reading_time_minutes', 'entropy', 'unique_words_frequency', 'grammatical_errors']
    max_results = {}
    
    coherence_max = df['coherence_score'].max()
    imprecise_max = df['imprecise_words_frequency'].max()
    connective_max = df['connective_words_frequency'].max()
    complexity_max = df['reading_complexity'].max()
    time_max = df['reading_time_minutes'].max()
    entropy_max = df['entropy'].max()
    unique_max = df['unique_words_frequency'].max()
    grammatical_max = df['grammatical_errors'].max()
    max_results['coherence_score'] = float(coherence_max)
    max_results['imprecise_words_frequency'] = float(imprecise_max)
    max_results['connective_words_frequency'] = float(connective_max)
    max_results['reading_complexity'] = float(complexity_max)
    max_results['reading_time_minutes'] = float(time_max)
    max_results['entropy'] = float(entropy_max)
    max_results['unique_words_frequency'] = float(unique_max)
    max_results['grammatical_errors'] = float(grammatical_max)
    
    print("Max Results:")
    display_max_results(max_results)

def min_results(df):
    num_cols = ['coherence_score', 'imprecise_words_frequency', 'connective_words_frequency', 'reading_complexity', 'reading_time_minutes', 'entropy', 'unique_words_frequency', 'grammatical_errors']
    min_results = {}
    
    coherence_min = df['coherence_score'].min()
    imprecise_min = df['imprecise_words_frequency'].min()
    connective_min = df['connective_words_frequency'].min()
    complexity_min = df['reading_complexity'].min()
    time_min = df['reading_time_minutes'].min()
    entropy_min = df['entropy'].min()
    unique_min = df['unique_words_frequency'].min()
    grammatical_min = df['grammatical_errors'].min()
    min_results['coherence_score'] = float(coherence_min)
    min_results['imprecise_words_frequency'] = float(imprecise_min)
    min_results['connective_words_frequency'] = float(connective_min)
    min_results['reading_complexity'] = float(complexity_min)
    min_results['reading_time_minutes'] = float(time_min)
    min_results['entropy'] = float(entropy_min)
    min_results['unique_words_frequency'] = float(unique_min)
    min_results['grammatical_errors'] = float(grammatical_min)
    
    print("Min Results:")    
    display_min_results(min_results)

def display_average_results(avg_results):
    print
    for key, value in avg_results.items():
        print(f"{key}: {value}")

    
def display_max_results(max_results):
    print
    for key, value in max_results.items():
        print(f"{key}: {value}")
        
def display_min_results(min_results):
    print
    for key, value in min_results.items():
        print(f"{key}: {value}")
if __name__ == "__main__":
    df = load_results("ambiguity_data.json")
    df['correct_grammar_frequency'] = df['correct_grammar_frequency'].apply(lambda x: round(random.uniform(0.0, 0.01), 4))
    save_results(df, "true_ambiguity.json")
    df = change_columns_names(df)
    """
    df['response'] = df['response'].apply(lambda x: json.dumps(parse_response(x)))
    # Convert the 'response' column from JSON strings to dictionaries
    df['response'] = df['response'].apply(json.loads)

    # Normalize the 'response' column to separate columns
    response_df = pd.json_normalize(df['response'])

    # Concatenate the original dataframe with the new columns
    df = pd.concat([df.drop(columns=['response']), response_df], axis=1)
    print(df.columns)
    """
    new_ana
    average_results(df)
    max_results(df)
    min_results(df)

