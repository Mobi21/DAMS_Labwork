import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm
import random

def load_results(filename="data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    return df


def save_results(results, filename="results.json"):
    results.to_json(filename, orient="records", indent=4)
    
    
def call_ollama(policy_text):
    prompt = """
        You are an expert in legal text analysis, specializing in privacy policies. Your task is to analyze the following privacy policy text and determine its level of ambiguity based on the criteria below.

        **Ambiguity Levels:**

        1. **Not Ambiguous (1):**


        2. **Somewhat Ambiguous (2):**


        3. **Ambiguous (3):**


        **Instructions:**

        - Carefully read the provided privacy policy text.
        - Determine the level of ambiguity based on the criteria above.
        - Do not include any additional commentary or analysis.
        - **Your response should strictly be in the following format:**

        Ambiguity_level:[value]


        The privacy policy text to analyze is provided below:
            {policy_text}
            \n
            
        Your response should only be:
        "Ambiguity_level":value(1-3)
        
        STRICT OUTPUT REQUIREMENT: Do not include any additional information, text, or explanation. 
        Your response must be in the format of Ambiguity_level:value(1-3). THERE SHOULD BE NO OTHER NUMBERS IN THE RESPONSE
    """

    try:
        response = ollama.generate(model="llama3.1", prompt= prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def call_ollama1(policy_text):
    prompt = """
    You are an expert in legal and policy analysis with a specialization in evaluating the clarity and transparency of privacy policies. 
    Your task is to analyze the full privacy policy text provided and assign an ambiguity level based on the rubric below. 
    The evaluation is based on how understandable, specific, and transparent the privacy policy is for users. Follow the instructions rigorously.

    ### Grading Rubric for Privacy Policy Ambiguity
    The privacy policy will be classified into one of three categories, with corresponding numeric levels:

    #### **1. NOT AMBIGUOUS**
    - The text is clear, explicit, and transparent throughout.
    - It uses well-defined terms, avoids vague language, and provides exhaustive descriptions of processes, rights, and responsibilities.
    - It explains:
        - **What data is collected** (e.g., personal information, cookies, IP addresses).
        - **How data is used** (e.g., for providing services, analytics, or advertising).
        - **Who data is shared with** (e.g., specific categories like vendors or partners).
        - **User rights** (e.g., deletion, access, correction).
        - **Data retention policies** (e.g., retention periods and reasons for keeping data).
    - **Examples**:
        - "We will only use your email address to send order confirmations and updates. Your email will not be shared with third parties."
        - "We retain your data for 30 days after account closure, unless legally required to keep it longer."
        - "You can access or delete your data by emailing privacy@company.com."
    - **Criteria**:
        - No room for interpretation or multiple meanings.
        - Every section is specific, complete, and easy to understand.

    #### **2. SOMEWHAT AMBIGUOUS**
    - The text provides some clarity but includes sections that could be interpreted in more than one way or use imprecise language.
    - The scope of terms or actions may lack full precision, and details may be incomplete or implied.
    - Examples:
        - "We may share your data with trusted partners to enhance your experience."
        - "We store data as long as it is necessary to provide services."
        - "Our vendors comply with applicable privacy laws."
    - **Criteria**:
        - Partial explanations of key practices.
        - Use of vague terms (e.g., "may," "necessary," "trusted partners").
        - Some clarity in certain sections, but ambiguity remains in others.

    #### **3. AMBIGUOUS**
    - The text is vague, unclear, or lacks specificity, leaving significant room for interpretation.
    - The policy fails to adequately explain key details such as:
        - What data is collected.
        - How data is used or shared.
        - User rights or data retention policies.
    - Frequent use of generic terms or legalese, making the policy difficult to understand.
    - **Examples**:
        - "Your data will be used for purposes deemed appropriate by the company."
        - "We follow applicable laws to protect your data."
        - "Information may be retained as needed."
    - **Criteria**:
        - Heavy reliance on ambiguous terms or phrases.
        - Key details are missing or obscured.
        - Little to no explanation of user rights or specific practices.

    ### Evaluation Instructions
    1. Carefully read the **entire privacy policy** provided.
    2. Analyze the policy as a whole based on the rubric:
        - Does the policy explicitly address all critical areas (data collection, use, sharing, rights, retention)?
        - Are vague terms clarified or defined (e.g., "trusted partners")?
        - Is the text clear and complete, or does it leave room for multiple interpretations?
    3. Assign an **overall numeric level of ambiguity** based on the most ambiguous portions of the policy:
        - **1** = NOT AMBIGUOUS
        - **2** = SOMEWHAT AMBIGUOUS
        - **3** = AMBIGUOUS

    ### Response Format
    Respond in the exact format below:
    ```
    Ambiguity_level:[1 | 2 | 3]
    ```

    ### Privacy Policy Text to Analyze
    {policy_text}

    ### Important Notes
    - Base your evaluation solely on the rubric above.
    - Do not include any commentary, explanation, or additional text in your response.
    - Ensure strict adherence to the response format, using only the specified levels (1, 2, or 3).
    """
    try:
        response = ollama.generate(model="llama3.1", prompt=prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None  
    
def parse_response(response):
    pattern = r'"Ambiguity_level":\s*([1-3])'
    match = re.search(r"1|2|3", response)
    
    if match:
        try:
            return float(match.group())
        except ValueError:
            logging.error("Error parsing response.")
            return None  
    else:
        logging.error("Error parsing response.")
        return None

def analysis(data):
    results = []
    for index, policy in tqdm(data.iterrows(), total=len(data), desc="Analyzing Privacy Policies"):
        policy_text = policy["policy_text"]
        
        found_ambiguity = False
        while not found_ambiguity:
            found_ambiguity = True
            response = call_ollama1(policy_text)
            response = parse_response(response.get("response"))
            if response is None:
                found_ambiguity = False
            else:
                policy["ambiguity_level"] = response
        
        results.append(policy)
            
    return pd.DataFrame(results)

def results(data):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for index, policy in data.iterrows():
        if policy["ambiguity_level"] == 1:
            count_1 += 1
        elif policy["ambiguity_level"] == 2:
            count_2 += 1
        elif policy["ambiguity_level"] == 3:
            count_3 += 1
            
    print(f"Total Policies: {len(data)}")
    print(f"Ambiguity Level 1: {count_1}")
    print(f"Ambiguity Level 2: {count_2}")
    print(f"Ambiguity Level 3: {count_3}")
   
if __name__ == "__main__":
   
    # Load the data
    data = load_results("final_data.json")


    # Analyze the privacy policies
    results = analysis(data)
    
    # Save the results
    save_results(results, "results.json")
    
    data = load_results("google_play_wayback.json")
    results = analysis(data)
    save_results(results, "wayback_results.json")

"""
    # Load the results
    result = load_results("ambiguity_true.json")
    results(result)
    
    result = load_results("ambiguity_false.json")
    results(result)
    """