# readability.py
import json
import re
import pandas as pd
import ollama
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_results(filename="final_data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_results(results, filename):
    pd.DataFrame(results).to_json(filename, orient="records", indent=4)

def call_ollama_with_library(prompt, prompt_helper2=""):
    model = "llama3.1"
    prompt_helper = (
        "\n YOUR ONLY OUTPUT SHOULD BE HE RESULT. DONT TELL ME HOW TO DO IT MYSELF OR GIVE ME A CODE ON HOW TO DO IT. give me my desired output that is all. Remember to use the correct name for the output name"
    )
    try:
        response = ollama.generate(model=model, prompt=prompt + prompt_helper + prompt_helper2)
        return response  # Response is already in JSON format
    except Exception as e:
        print(f"Error: {e}")
        return None

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

def parse_response(response_text, text):
    # This parser uses a simple regex for demonstration.
    patterns = {
        'coherence_score': r'"coherence_score":\s*([0-9.]+)',
        'imprecise_words_frequency': r'"imprecise_words_frequency":\s*([0-9.]+)',
        'connective_words_frequency': r'"connective_words_frequency":\s*([0-9.]+)',
        'reading_complexity': r'"reading_complexity":\s*([0-9.]+)',
        'reading_time': r'"reading_time":\s*([0-9.]+)',
        'entropy': r'"entropy":\s*([0-9.]+)',
        'unique_words_frequency': r'"unique_words_frequency":\s*([0-9.]+)',
        'grammatical_errors': r'"grammatical_errors":\s*([0-9.]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None

def process_policy(policy):
    updated_policy = policy.copy()
    policy_text = updated_policy["policy_text"]
    metrics = {
        "coherence_score": get_coherence_score,
        "imprecise_words_frequency": get_imprecise_words_frequency,
        "connective_words_frequency": get_connective_words_frequency,
        "reading_complexity": get_reading_complexity,
        "reading_time": get_reading_time,
        "entropy": get_entropy,
        "unique_words_frequency": get_unique_words_frequency,
        "grammatical_errors": get_grammatical_errors
    }
    log_entry = {}
    for metric, func in metrics.items():
        found = False
        count = 0
        metric_response = None
        while not found and count < 7:
            count += 1
            metric_response = func(policy_text)
            response_text = metric_response.get("response") if metric_response and hasattr(metric_response, "get") else ""
            parsed = parse_response(response_text, policy_text)
            if parsed is not None or count >= 7:
                updated_policy[metric] = parsed
                found = True
        log_entry[metric] = {"response": metric_response, "retries": count}
    return updated_policy, log_entry

def run_tests(output_dir="results"):
    df1 = load_results("final_data.json")
    result1 = []
    logs1 = []
    for policy in tqdm(df1.to_dict("records"), total=len(df1), desc="Readability (final_data.json)"):
        res, log = process_policy(policy)
        result1.append(res)
        logs1.append(log)
    df2 = load_results("google_play_wayback.json")
    result2 = []
    logs2 = []
    for policy in tqdm(df2.to_dict("records"), total=len(df2), desc="Readability (google_play_wayback.json)"):
        res, log = process_policy(policy)
        result2.append(res)
        logs2.append(log)
    final_results = result1 + result2
    final_logs = logs1 + logs2
    save_results(final_results, f"{output_dir}/readability_final_results.json")
    save_results(final_logs, f"{output_dir}/readability_final_logs.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_tests("results")
