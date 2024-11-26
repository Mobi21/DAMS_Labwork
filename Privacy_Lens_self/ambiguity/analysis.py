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

def call_ollama_with_library(prompt, prompt_helper2=""):
    model = "llama3.1"
    prompt_helper = (
        "\n YOUR ONLY OUTPUT SHOULD BE HE RESULT. DONT TELL ME HOW TO DO IT MYSELF OR GIVE ME A CODE ON HOW TO DO IT. give me my desired output that is all. Remember to use the correct name for the output name"
    )
    try:
        # Use Ollama library to interact with the model
        response = ollama.generate(model=model, prompt=prompt + prompt_helper + prompt_helper2)
        return response  # Response is already in JSON format
    except Exception as e:
        print(f"Error: {e}")
        return None

# Define functions to call for each metric
def get_coherence_score(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to calculate the CoherenceScore of a given privacy policy text. 
    The CoherenceScore quantifies how comprehensible and logically connected the text is. 
    Regarding readability, a high coherence score implies a well-structured and clear flow of
    ideas, enhancing readability. Conversely, a low score might indicate
    a disjointed or unclear thought progression, making the text harder
    to comprehend.Imagine it was an average user reading the policy text.
    The coherence_score should be from a range of 0 to 1. 

    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "coherence_score": value(0-1)
    """
    return call_ollama_with_library(prompt)




def get_imprecise_words_frequency(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to analyze a privacy policy text and calculate the frequency of imprecise words. 
    Imprecise words include vague terms like "commonly," "normally," "generally," and similar ambiguous phrases. 
    The frequency is calculated as:

    imprecise_words_frequency = (Total Imprecise Words / Total Words in Text)

    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "imprecise_words_frequency": value(0-1)
    """
    return call_ollama_with_library(prompt)




def get_connective_words_frequency(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to analyze a privacy policy text and calculate the frequency of connective words. 
    Connective words include terms like "and," "but," "then," "therefore," and similar linking phrases.
    The frequency is calculated as:

    connective_words_frequency = (Total Connective Words / Total Words in Text)

    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "connective_words_frequency": value(0-1)
    """
    return call_ollama_with_library(prompt)




def get_reading_complexity(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to calculate the ReadingComplexity of a given privacy policy text using the Flesch-Kincaid Grade Level . 
    FKGL measures the educational level required to understand the text. Use this formula:

    FKGL = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59

    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "reading_complexity": value
    """
    return call_ollama_with_library(prompt)



def get_reading_time(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to calculate the ReadingTime of a given privacy policy text. 
    Assume an average reader is reading the policy text. Calculate the reading time in minutes based on this.

    (generally time range should be roughly in 1-140mins)
    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "reading_time": value
    """
    return call_ollama_with_library(prompt)



def get_entropy(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to calculate the Entropy of a given privacy policy text. 
    Entropy quantifies the uncertainty and complexity of the text. Use Shannon's entropy formula:

    Entropy = -Σ(p(x) * log2(p(x)))

    where p(x) is the probability of each character in the text. 

    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "entropy": value
    """
    return call_ollama_with_library(prompt)



def get_unique_words_frequency(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to analyze a privacy policy text and calculate the frequency of unique words. 
    Unique words Frequency indicates the diversity of vocabulary used in the text. Every new word is counted a unique word. This isn't about how complex the word is but how many different words are used.
    The unique_words_frequency should be a value from 0-1.
    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "unique_words_frequency": value
    """
    return call_ollama_with_library(prompt)




def get_grammatical_errors(policy_text):
    prompt = """
    You are an advanced NLP model. Your task is to analyze a privacy policy text and identify all sentences that contain grammatical errors. Count the amount of grammatical errors in the text.Your putput should be the frequency of grammatical errors between 0.000 to 0.01.

    Input: 
    The privacy policy text to analyze is provided below:
        {policy_text}
        \n
    Output: 
    Return the result as:
    "grammatical_errors": value
    """
    return call_ollama_with_library(prompt)

"""
def analysis(data):
    results = []
    # Wrap data.iterrows() with tqdm for a progress bar
    for _, policy in tqdm(data.iterrows(), total=len(data), desc="Analyzing policies"):
        valid = False
        while(valid == False):
            valid = True
            policy_text = policy["policy_text"]
            result = {
                "coherence_score": get_coherence_score(policy_text),
                "imprecise_words_frequency": get_imprecise_words_frequency(policy_text),
                "connective_words_frequency": get_connective_words_frequency(policy_text),
                "reading_complexity": get_reading_complexity(policy_text),
                "reading_time": get_reading_time(policy_text),
                "entropy": get_entropy(policy_text),
                "unique_words_frequency": get_unique_words_frequency(policy_text),
                "grammatical_errors": get_grammatical_errors(policy_text),
            }
            policy["coherence_score"] = parse_response(result["coherence_score"].get('response'), policy_text)
            policy["imprecise_words_frequency"] = parse_response(result["imprecise_words_frequency"].get('response'), policy_text)
            policy["connective_words_frequency"] = parse_response(result["connective_words_frequency"].get('response'), policy_text)
            policy["reading_complexity"] = parse_response(result["reading_complexity"].get('response'), policy_text)
            policy["reading_time"] = parse_response(result["reading_time"].get('response'), policy_text)
            policy["entropy"] = parse_response(result["entropy"].get('response'), policy_text)
            policy["unique_words_frequency"] = parse_response(result["unique_words_frequency"].get('response'), policy_text)
            policy["grammatical_errors"] = parse_response(result["grammatical_errors"].get('response'), policy_text)
            
            if None in result.values():
                valid = False
            else:
                results.append(policy)
    
    updated_data = pd.DataFrame(results)
    return updated_data
"""
def analysis(data):
    results = []
    # Wrap data.iterrows() with tqdm for a progress bar
    for _, policy in tqdm(data.iterrows(), total=len(data), desc="Analyzing policies"):
        policy_text = policy["policy_text"]
        
        found_coherence_score = False
        while not found_coherence_score:
            found_coherence_score = True
            response = get_coherence_score(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_coherence_score = False
            else:
                policy["coherence_score"] = response
                
        found_imprecise_words_frequency = False
        while not found_imprecise_words_frequency:
            found_imprecise_words_frequency = True
            response = get_imprecise_words_frequency(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_imprecise_words_frequency = False
            else:
                policy["imprecise_words_frequency"] = response
                
        found_connective_words_frequency = False
        while not found_connective_words_frequency:
            found_connective_words_frequency = True
            response = get_connective_words_frequency(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_connective_words_frequency = False
            else:
                policy["connective_words_frequency"] = response
                
        found_reading_complexity = False
        while not found_reading_complexity:
            found_reading_complexity = True
            response = get_reading_complexity(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_reading_complexity = False
            else:
                policy["reading_complexity"] = response
        
        found_reading_time = False
        while not found_reading_time:
            found_reading_time = True
            response = get_reading_time(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_reading_time = False
            else:
                policy["reading_time"] = response
                
        found_entropy = False
        while not found_entropy:
            found_entropy = True
            response = get_entropy(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_entropy = False
            else:
                policy["entropy"] = response
                
        found_unique_words_frequency = False
        while not found_unique_words_frequency:
            found_unique_words_frequency = True
            response = get_unique_words_frequency(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_unique_words_frequency = False
            else:
                policy["unique_words_frequency"] = response
                
        found_grammatical_errors = False
        while not found_grammatical_errors:
            found_grammatical_errors = True
            response = get_grammatical_errors(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_grammatical_errors = False
            else:
                policy["grammatical_errors"] = response
                


        results.append(policy)
    updated_data = pd.DataFrame(results)
    return updated_data

def selective_analysis(data):
    results = []
    for _, policy in tqdm(data.iterrows(), total=len(data), desc="Analyzing policies"):
        policy_text = policy["policy_text"]
        
        found_coherence_score = False
        while not found_coherence_score:
            found_coherence_score = True
            response = get_coherence_score(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_coherence_score = False
            else:
                policy["coherence_score"] = response
                

        found_reading_time = False
        while not found_reading_time:
            found_reading_time = True
            response = get_reading_time(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_reading_time = False
            else:
                policy["reading_time"] = response
                
        found_unique_words_frequency = False
        while not found_unique_words_frequency:
            found_unique_words_frequency = True
            response = get_unique_words_frequency(policy_text)
            response = parse_response(response.get('response'), policy_text)
            if response is None:
                found_unique_words_frequency = False
            else:
                policy["unique_words_frequency"] = response
                
                


        results.append(policy)
    updated_data = pd.DataFrame(results)
    return updated_data

def save_results(results, filename="results.json"):
    results.to_json(filename, orient="records", indent=4)

def parse_response(response, text):
    metrics = {}
    patterns = {
        'coherence_score': r'"coherence_score":\s*([0-9.]+)',
        'imprecise_words_frequency': r'"imprecise_words_frequency":\s*([0-9.]+)',
        'connective_words_frequency': r'"connective_words_frequency":\s*([0-9.]+)',
        'reading_complexity': r'"reading_complexity":\s*([0-9.]+)',
        'reading_time': r'"reading_time":\s*([0-9.]+)',
        'entropy': r'"entropy":\s*([0-9.]+)',
        'unique_words_frequency': r'"unique_words_frequency":\s*([0-9.]+)',
        'grammatical_errors': r'"grammatical_errors":\s*([0-9.]+)',
    }
    found = 0
    match = False
    key = None
    for keys, pattern in patterns.items():
        matches = re.search(pattern, response)
        if matches:
            match = matches
            key = keys
            
            
    if match:
        found += 1
        value = match.group(1)
        if key == 'reading_time':
            try:
                metrics['reading_time'] = float(value)
            except ValueError:
                metrics['reading_time'] = None
                logging.warning(f"Invalid reading_time value: {value}")
        elif key == 'entropy':
            try:
                metrics['entropy'] = float(value)
            except ValueError:
                metrics['entropy'] = None
                logging.warning(f"Invalid grammatical_errors value: {value}")
        elif key == 'reading_complexity':
            try:
                metrics['reading_complexity'] = float(value)
            except ValueError:
                metrics['reading_complexity'] = None
                logging.warning(f"Invalid reading_complexity value: {value}")
        elif key == 'coherence_score':
            try:
                metrics['coherence_score'] = float(value)
            except ValueError:
                metrics['coherence_score'] = None
                logging.warning(f"Invalid coherence_score value: {value}")
        elif key == 'grammatical_errors':
            try:
                metrics['grammatical_errors'] = float(value)
            except ValueError:
                metrics['grammatical_errors'] = None
                logging.warning(f"Invalid grammatical_errors value: {value}")
        elif key == 'imprecise_words_frequency':
            try:
                metrics['imprecise_words_frequency'] = float(value)
            except ValueError:
                metrics['imprecise_words_frequency'] = None
                logging.warning(f"Invalid imprecise_words_frequency value: {value}")       
        elif key == 'connective_words_frequency':
            try:
                metrics['connective_words_frequency'] = float(value)
            except ValueError:
                metrics['connective_words_frequency'] = None
                logging.warning(f"Invalid connective_words_frequency value: {value}")
        elif key == 'unique_words_frequency':
            try:
                metrics['unique_words_frequency'] = float(value)
            except ValueError:
                metrics['unique_words_frequency'] = None
                logging.warning(f"Invalid unique_words_frequency value: {value}")
                
        else:
            try:
                value = "[" + value.strip(", ").strip("[]") + "]"  # Wrap in brackets
                metrics[key] = json.loads(value)
                print(len(metrics[key]))
                metrics[key] = int(len(metrics[key]))/ word_count(text)
            except json.JSONDecodeError:
                metrics[key] = None
                logging.warning(f"Invalid list value for {key}: {value}")
    else:
        logging.warning(f"Nothing found in response.")
        metrics[key] = None
    
    return metrics[key]

def word_count(text):
    return len(text.split())
def load_results(filename="data.json"):
    with open(filename, encoding='utf-8') as file:
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
    time_average = df['reading_time'].median()
    entropy_average = df['entropy'].median()
    unique_average = df['unique_words_frequency'].median()
    grammatical_average = df['grammatical_errors'].median()
    avg_results['coherence_score'] = float(coherence_average)
    avg_results['imprecise_words_frequency'] = float(imprecise_average)
    avg_results['connective_words_frequency'] = float(connective_average)
    avg_results['reading_complexity'] = float(complexity_average)
    avg_results['reading_time'] = float(time_average)
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
    time_max = df['reading_time'].max()
    entropy_max = df['entropy'].max()
    unique_max = df['unique_words_frequency'].max()
    grammatical_max = df['grammatical_errors'].max()
    max_results['coherence_score'] = float(coherence_max)
    max_results['imprecise_words_frequency'] = float(imprecise_max)
    max_results['connective_words_frequency'] = float(connective_max)
    max_results['reading_complexity'] = float(complexity_max)
    max_results['reading_time'] = float(time_max)
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
    time_min = df['reading_time'].min()
    entropy_min = df['entropy'].min()
    unique_min = df['unique_words_frequency'].min()
    grammatical_min = df['grammatical_errors'].min()
    min_results['coherence_score'] = float(coherence_min)
    min_results['imprecise_words_frequency'] = float(imprecise_min)
    min_results['connective_words_frequency'] = float(connective_min)
    min_results['reading_complexity'] = float(complexity_min)
    min_results['reading_time'] = float(time_min)
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
        
def change_data(main_data, new_data, change_columns):
    for column in change_columns:
        main_data[column] = new_data[column]
    return main_data
def change_columns_names(df):
    df = df.rename(columns={"url": "manufacturer", "Coherance Score": "coherence_score", "Imprecise Words": "imprecise_words_frequency", "Connective Words": "connective_words_frequency", "flesch_kincaid_grade_level": "reading_complexity", "Reading_Time (Min)": "reading_time_minutes", "Entropy": "entropy", "Unique_words": "unique_words_frequency", "correct_grammar_frequency": "grammatical_errors", "Reading_Time (Min)": "reading_time", "Unique Words": "unique_words_frequency"})
    return df

def combine_data(df, df1, columnsss):
    for column in columnsss:
        df[column] = df1[column]
    return df

def combine_df(df1, df2):
    df = pd.concat([df1, df2])
    return df
if __name__ == "__main__":
    """
    df1 = load_results("ambiguity_data.json")
    df1 = change_columns_names(df1)
    
    df = load_results("final_results.json")
    df = change_columns_names(df)

    
    
    
    df = combine_data(df, df1, ['coherence_score', 'unique_words_frequency', 'grammatical_errors'])

    
    

    df1 = load_results("ambiguity_wayback.json")
    combined_df = combine_df(df, df1)
    average_results(combined_df)
    max_results(combined_df)
    min_results(combined_df)
    """
    df = load_results('ambiguity_data.json')
    df1 = load_results('wayback_true_amb.json')
    df = combine_df(df, df1)
    df = change_columns_names(df)
    average_results(df)
    min_results(df)
    max_results(df)
