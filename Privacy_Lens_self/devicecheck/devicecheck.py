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

device_list = ["Smart Speaker", "Smart Thermostat", "Smart Camera", "Smart Lock", "Smart Fitness Tracker", "Smart Light", "Smart Doorbell", "Smart Alarm System", "Smart TV", "Smart Scale", "Smart Home Device", "Smart Air Purifier", "Smart Sensor", "Smart Watch", "Smart Monitor", "Smart Security", "Smart Health Tracker", "Smart Refrigerator", "Smart Location Tracker", "Smart Entertainment Device", "Smart Connected Vehicle", "Smart Networking", "Smart Fitness Equipment", "Smart Mount", "Smart Projector", "Smart Body Scanners", "Smart Gaming"]

def load_results(filename="data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if filename == "results.json":
        df = df[['manufacturer', 'response']]
    return df

def save_results(results, filename="results.json"):
    df = pd.DataFrame(results)
    df.to_json(filename, orient="records", indent=4)

def check_device_in_policy_with_ollama5(policy_text, model="llama3.1"):
    prompt = """
    You are a highly skilled text analysis assistant specializing in legal and policy documents. Your task is to analyze the provided company policy text and determine whether it **explicitly mentions any smart device manufactured by the company and if the policy explicitly applies to that device.**
    
    ### Evaluation Criteria
    To determine whether the policy explicitly mentions a smart device, follow these detailed steps:
    
    #### 1. **Explicit Mention of a Smart Device**
    - A device is explicitly mentioned if the text specifically names a smart device category or product manufactured by the company (e.g., "Smart Speaker," "Smart Camera").
    - Vague references such as "our devices" or "connected products" without specific naming are not considered explicit.
    - **Examples of Explicit Mentions:**
      - "Our privacy policy applies to Smart Cameras and Smart Speakers."
      - "This policy covers the Smart Thermostat and other home automation devices."

    #### 2. **Application of Policy to the Device**
    - The policy must clearly state that it applies to the named device(s). This may include statements like:
      - "Data collected from our Smart Thermostat is governed by this policy."
      - "This privacy policy applies to all Smart Doorbells manufactured by our company."
    - Ambiguous or implied applications, such as "applies to all our products," are not sufficient without further clarification.

    #### 3. **Exclusions**
    - If the policy mentions a smart device but does not explicitly state that it is covered, it should not be marked as covered.
    - Examples of exclusions:
      - "Smart Lights may collect data, but they are not covered by this policy."
      - "We offer various connected products, but this policy does not govern their use."

    ### Response Requirements
    - If the policy explicitly mentions and applies to a smart device, return:
      ```
      Device: [Smart Device Name]
      ```
      (e.g., "Device: Smart Speaker")
    - If the policy does not mention or does not explicitly apply to any smart device, return:
      ```
      Device: None
      ```

    ### Policy Text to Analyze
    {policy_text}

    ### Strict Output Instructions
    - **Do not provide any commentary, explanation, or additional information.**
    - Your response must only consist of:
      - `Device: [Smart Device Name]` (if a specific device is mentioned and covered).
      - `Device: None` (if no device is mentioned or covered).
    - Responses that deviate from this format will be invalid.

    ### Important Notes
    - If the policy contains multiple devices, return only the first one explicitly mentioned and covered.
    - Base your evaluation solely on the text provided, adhering to the criteria above.
    """
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_response3(response):
    try:
        if isinstance(response, dict):
            response_text = response.get("response", "")
        else:
            response_text = response
        response_text = response_text.strip()
        if response_text.lower().startswith('device:'):
            device_info = response_text[len('Device:'):].strip()
            device_info = device_info.split('\n')[0].strip()
            if device_info.lower() == 'none':
                return None
            else:
                return device_info
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_device_policy(row):
    manufacturer = row['manufacturer']
    policy_text = row['policy_text']
    response_dict = check_device_in_policy_with_ollama5(policy_text)
    log_entry = {"manufacturer": manufacturer, "policy_text": policy_text}
    if response_dict:
        log_entry.update(response_dict)
    response_text = response_dict.get('response') if response_dict else ""
    parsed_device = parse_response3(response_text)
    devicecheck = True if parsed_device and len(parsed_device) > 0 else False
    result = {
        "manufacturer": manufacturer,
        "devices": parsed_device if parsed_device else "",
        "DeviceCheck": devicecheck,
        "policy_text": policy_text
    }
    return result, log_entry

def check_all_devices_in_policy(df, devices_list):
    results = []
    logs = []
    records = df.to_dict("records")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for res, log in tqdm(executor.map(process_device_policy, records), total=len(records), desc="Checking Devices in Policies"):
            results.append(res)
            logs.append(log)
    return pd.DataFrame(results), pd.DataFrame(logs)

def device_count(df):
    count = 0
    for _, row in df.iterrows():
        if row['DeviceCheck'] == True:
            count += 1
    print(f"Number of policies mentioning a device: {count}")
    return count

def combine_df(df1, df2):
    df = pd.concat([df1, df2])
    return df

if __name__ == "__main__":
    # For example, using google_play_wayback.json:
    
        # Process final_data.json
    data1 = load_results("final_data.json")
    results1, logs1 = check_all_devices_in_policy(data1, device_list)
    
    # Process google_play_wayback.json
    data2 = load_results("google_play_wayback.json")
    results2, logs2 = check_all_devices_in_policy(data2, device_list)
    
    # Combine both datasets so that both final files have the same amount of rows
    final_results = pd.concat([results1, results2], ignore_index=True)
    final_logs = pd.concat([logs1, logs2], ignore_index=True)
    
    # Save the final combined results and logs
    save_results(final_results, "final_results_final.json")
    save_results(final_logs, "data_logs.json")
    
    

