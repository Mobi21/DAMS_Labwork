import json
import pandas as pd
import logging
from tqdm import tqdm
import re
import ollama  # Ensure Ollama is properly installed and configured
from concurrent.futures import ThreadPoolExecutor

# Load and save functions
def load_results(filename="data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_results(results, filename="results.json"):
    results.to_json(filename, orient="records", indent=4)

# Keyword extraction function
def call_ollama_with_library_for_keywords(text, category, category_description, model="llama3.1"):
    prompt = f"""
    You are a highly skilled text analysis assistant specializing in extracting specific keywords from provided text.
    Your task is to identify all occurrences of keywords related to a single category and return them as a list.

    ### Instructions
    1. **Keyword Identification**:
       - Identify all keywords that align with the given category description.
       - Use the category description to determine loosely matching keywords.
       - Include all occurrences of matching keywords, including duplicates.

    2. **Category and Description**:
       - **Category**: "{category}"
       - **Description**: "{category_description}"

    3. **Output Requirements**:
       - Your output must be a single structured JSON array containing all matching keywords, e.g.,:
         ```
         ["keyword1", "keyword2", "keyword3", ...]
         ```
       - If no matching keywords are found, return an empty array: `[]`.
       - Provide no other text or commentary outside the JSON array.

    ### Text to Analyze
    {text}

    ### Important Notes
    - Base your analysis solely on the category description and the text provided.
    - Include duplicates in the JSON array.
    - Ensure the output contains only the JSON array.
    """
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

# Parsing function for Ollama response
def parse_keywords_from_text(text):
    """
    Extract the text within double quotes inside square brackets.
    
    Returns the count of keywords found.
    """
    try:
        # Regular expression to match strings inside double quotes
        matches = re.findall(r'"(.*?)"', text)
        return len(matches)
    except Exception as e:
        logging.error(f"Error while parsing text: {e}")
        return 0

# Helper function to process a single policy for keyword extraction
def process_policy_keywords(policy, categories):
    result = {"manufacturer": policy["manufacturer"]}
    logs = []  # list to hold log entries for each category
    policy_text = policy["policy_text"]
    for category, description in categories.items():
        retries = 0
        valid = False
        response_dict = None
        keywords = 0
        while not valid and retries < 5:
            retries += 1
            response_dict = call_ollama_with_library_for_keywords(policy_text, category, description)
            response_text = response_dict.get("response") if response_dict else ""
            keywords = parse_keywords_from_text(response_text)
            # Accept even a zero count as a valid result
            if keywords is not None:
                valid = True
        if not valid:
            logging.warning(f"Failed to extract keywords for category {category}.")
        result[category] = keywords
        # Log entry capturing full response info along with metadata
        log_entry = {
            "manufacturer": policy["manufacturer"],
            "policy_text": policy_text,
            "category": category,
            "retries": retries
        }
        if response_dict:
            log_entry.update(response_dict)
        logs.append(log_entry)
    return result, logs

# Data collection function with concurrency
def collect_keyword_data(data, categories):
    results = []
    all_logs = []
    records = data.to_dict("records")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for res, log in tqdm(executor.map(lambda record: process_policy_keywords(record, categories), records),
                             total=len(records), desc="Keyword Collection"):
            results.append(res)
            all_logs.extend(log)
    return pd.DataFrame(results), pd.DataFrame(all_logs)

# Analysis functions
def keyword_summary(df):
    for column in df.columns[1:]:
        print(f"Category: {column}")
        print(f" - Total Keywords: {df[column].sum()}")
        print(f" - Average Keywords: {df[column].mean():.2f}")
        print(f" - Max Keywords: {df[column].max()}")
        print(f" - Null Count: {df[column].isnull().sum()}")

# Combine dataframes
def combine_df(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

# Main execution
if __name__ == "__main__":
    # Define categories and descriptions
    categories = {
        "do_not_track": "Keywords related to 'Do Not Track' functionality or user requests to minimize tracking.",
        "data_security": "Keywords related to data protection, encryption, and security measures.",
        "first_party_collection": "Keywords about data collected directly by the company.",
        "third_party_collection": "Keywords about data sharing or collection by third parties.",
        "opt_out": "Keywords about users opting out of data collection or specific services.",
        "user_choice": "Keywords about user preferences, control, or decision-making regarding data.",
        "data": "Generic terms related to data or information, including personal and demographic data.",
        "legislation": "Keywords referencing privacy laws, regulations, or compliance standards.",
        "access_edit_delete": "Keywords about accessing, editing, or deleting personal data.",
        "policy_change": "Keywords about modifications, updates, or changes to privacy policies."
    }

    
    
    
    # Process first dataset
    df = load_results('final_data.json')
    results_df, logs_df = collect_keyword_data(df, categories)

    # Process second dataset
    df1 = load_results('google_play_wayback.json')
    results_df1, logs_df1 = collect_keyword_data(df1, categories)

    final_results = pd.concat([results_df, results_df1], ignore_index=True)
    final_logs = pd.concat([logs_df, logs_df1], ignore_index=True)
    
    # Save the final combined results and logs
    save_results(final_results, "keyword_results.json")
    save_results(final_logs, "data_logs.json")
    # Optionally, you can print a summary
    keyword_summary(results_df)
