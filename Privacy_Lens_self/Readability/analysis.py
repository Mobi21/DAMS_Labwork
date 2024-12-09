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
    model = "llama3.1:70b"
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
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in evaluating privacy policies. Your task is to assess the coherence of the provided privacy policy text. 
    Coherence measures how logically connected and comprehensible the text is, particularly for an average user. A well-structured text with a clear flow of ideas will receive a high coherence score, while a poorly organized text with unclear or disjointed thought progression will score lower.

    ### Instructions
    1. **Analyze the structure and flow of ideas**:
       - Check whether sentences are logically connected and transitions are smooth.
       - Assess whether the text is easy to follow and understand without requiring re-reading.
    2. **Evaluate from an average user's perspective**:
       - Consider how an average user (non-expert in legal or technical terms) would perceive the text.
    3. **Assign a coherence score**:
       - Use a scale from 0 to 1:
         - **0**: Completely incoherent, with no logical flow or readability.
         - **1**: Perfectly coherent, with clear, logical, and easy-to-follow structure.
       - Intermediate scores should reflect partial coherence or specific weaknesses in logical flow.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "coherence_score": value(0-1)
    ```
    """
    return call_ollama_with_library(prompt)


def get_imprecise_words_frequency(policy_text):
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in identifying imprecise language in privacy policies. Your task is to calculate the frequency of imprecise words within the provided text. 
    Imprecise words include vague terms such as "commonly," "normally," "generally," and similar ambiguous phrases, which reduce the clarity of the policy.

    ### Instructions
    1. **Identify all occurrences of imprecise words**:
       - Look for vague or non-specific terms that lack clear definitions or boundaries.
       - Include commonly used ambiguous phrases such as "as needed" or "may."
    2. **Calculate the frequency**:
       - Use the formula:
         ```
         imprecise_words_frequency = (Total Imprecise Words / Total Words in Text)
         ```
       - Ensure the result is a value between 0 and 1.
    3. **Focus on user comprehension**:
       - Consider the impact of imprecise words on the average user's ability to understand the policy.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "imprecise_words_frequency": value(0-1)
    ```
    """
    return call_ollama_with_library(prompt)


def get_connective_words_frequency(policy_text):
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in analyzing the structure of privacy policies. Your task is to calculate the frequency of connective words within the provided text. 
    Connective words include terms like "and," "but," "then," "therefore," and similar linking phrases that indicate relationships between ideas.

    ### Instructions
    1. **Identify all occurrences of connective words**:
       - Include commonly used conjunctions and transitional phrases.
       - Look for terms that enhance the logical flow of the text.
    2. **Calculate the frequency**:
       - Use the formula:
         ```
         connective_words_frequency = (Total Connective Words / Total Words in Text)
         ```
       - Ensure the result is a value between 0 and 1.
    3. **Evaluate the role of connective words**:
       - Assess how effectively they contribute to the readability and logical structure of the policy.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "connective_words_frequency": value(0-1)
    ```
    """
    return call_ollama_with_library(prompt)


def get_reading_complexity(policy_text):
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in assessing readability metrics. Your task is to calculate the reading complexity of the provided privacy policy text using the Flesch-Kincaid Grade Level (FKGL). 
    FKGL measures the educational level required to understand the text.

    ### Instructions
    1. **Use the FKGL formula**:
       ```
       FKGL = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
       ```
    2. **Analyze the text for the required components**:
       - Count the total number of words, sentences, and syllables.
       - Plug the values into the formula to compute the FKGL.
    3. **Interpret the result**:
       - Higher FKGL values indicate more complex texts requiring higher educational levels.
       - Lower FKGL values indicate simpler texts.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "reading_complexity": value
    ```
    """
    return call_ollama_with_library(prompt)


def get_reading_time(policy_text):
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in estimating reading times. Your task is to calculate the approximate reading time of the provided privacy policy text in minutes. 

    ### Instructions
    1. **Use average reading speed**:
       - Assume an average reading speed of approximately 200 words per minute.
    2. **Calculate the reading time**:
       - Use the formula:
         ```
         reading_time = total_words / 200
         ```
       - Round the result to the nearest whole number.
    3. **Output the reading time in minutes**:
       - Ensure the time falls within the range of 1 to 140 minutes.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "reading_time": value
    ```
    """
    return call_ollama_with_library(prompt)


def get_entropy(policy_text):
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in quantifying textual complexity. Your task is to calculate the entropy of the provided privacy policy text. 
    Entropy measures the level of uncertainty and complexity within the text, based on character distribution.

    ### Instructions
    1. **Use Shannon's entropy formula**:
       ```
       Entropy = -Σ(p(x) * log2(p(x)))
       ```
       - p(x) is the probability of each character in the text.
    2. **Analyze the character distribution**:
       - Count the frequency of each character.
       - Compute the probabilities and calculate entropy.
    3. **Output the entropy value**:
       - Higher entropy indicates more complexity and unpredictability.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "entropy": value
    ```
    """
    return call_ollama_with_library(prompt)


def get_unique_words_frequency(policy_text):
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in analyzing vocabulary diversity. Your task is to calculate the frequency of unique words in the provided privacy policy text. 
    Unique words frequency reflects the variety of vocabulary used.

    ### Instructions
    1. **Identify all unique words**:
       - Count every distinct word in the text.
    2. **Calculate the frequency**:
       - Use the formula:
         ```
         unique_words_frequency = (Total Unique Words / Total Words in Text)
         ```
       - Ensure the result is a value between 0 and 1.
    3. **Interpret the result**:
       - Higher frequency indicates greater vocabulary diversity.
       - Lower frequency suggests repetition or limited vocabulary.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "unique_words_frequency": value
    ```
    """
    return call_ollama_with_library(prompt)


def get_grammatical_errors(policy_text):
    prompt = f"""
    You are a highly skilled text analysis assistant with expertise in grammar analysis. Your task is to calculate the frequency of grammatical errors in the provided privacy policy text. 

    ### Instructions
    1. **Identify all sentences containing grammatical errors**:
       - Detect any deviations from standard grammar rules.
    2. **Calculate the frequency**:
       - Use the formula:
         ```
         grammatical_errors = (Total Grammatical Errors / Total Words in Text)
         ```
       - Ensure the result is a value between 0 and 1.
    3. **Interpret the result**:
       - Higher frequency indicates poorer grammar quality.
       - Lower frequency indicates better grammar quality.

    ### Input
    Privacy policy text:
    {policy_text}

    ### Output Format
    Your response must strictly follow this format:
    ```
    "grammatical_errors": value
    ```
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
        """        
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
            """    
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
    def average_results(df):
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
        print()
        for key, value in avg_results.items():
            print(f"{key}: {value}")

        
    def display_max_results(max_results):
        print()
        for key, value in max_results.items():
            print(f"{key}: {value}")
            
    def display_min_results(min_results):
        print()
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

        df = combine_data(df, df1, ['coherence_score', 'unique_words_frequency'])
        """
        
        print("TRUE DATA")
        df1 = load_results("readability_true.json")
        average_results(df1)
        max_results(df1)
        min_results(df1)
        
        print("FALSE DATA")
        df2 = load_results("readability_false.json")
        average_results(df2)
        max_results(df2)
        min_results(df2)
        
        """
        df = load_results('ambiguity_data.json')
        df1 = load_results('wayback_true_amb.json')
        df = combine_df(df, df1)
        df = change_columns_names(df)
        average_results(df)
        min_results(df)
        max_results(df)
        """
        
        """
        df = load_results('final_results.json')
        df1 = load_results('ambiguity_wayback.json')
        df = combine_df(df, df1)
        average_results(df)        """

