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
    

def check_device_in_policy_with_ollama(policy_text, device, model="llama3.1"):
    # Define the prompt for analyzing the policy text for the specified device
    prompt = (
        "You are a precise and logical text analysis assistant. Analyze the following policy text and determine if "
        "A specified device in the category or the category itself is mentioned anywhere in the text. the wording doesn't need to be exactly the same but it must mention a device or type of device. Return only 'True' if the device is mentioned "
        "and 'False' if it is not. Provide no additional output, explanation, or text other than 'True' or 'False'.\n\n"
        
        "Device Category to check: " + device + "\n\n"
        
        "Policy Text to Analyze:\n"
        + policy_text +
        "\n\n"
        
        "Your response should only be:\n"
        "- True: if the category or device of the category is mentioned in the policy text.\n"
        "- False: if the category or device of the category is not mentioned in the policy text.\n\n"
        
        "STRICT OUTPUT REQUIREMENT: Do not include any additional information, text, or explanation. Your response must "
        "be a single word: either 'True' or 'False'."
    )

    try:
        # Use Ollama library to interact with the model
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Ensure response is properly stripped to avoid whitespace issues
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_device_in_policy_with_ollama(policy_text, devices_list, model="llama3.1"):
    # Define the prompt for analyzing the policy text for the specified devices
    prompt = (
        "You are a precise and logical text analysis assistant. Analyze the following policy text and determine if "
        "any of the specified device categories are mentioned anywhere in the text. The wording doesn't need to be exactly the same but it must mention a device or type of device corresponding to one of the categories. Return the device category that is mentioned "
        "or 'False' if none are mentioned. Provide no additional output, explanation, or text other than the device category or 'False'.\n\n"
        
        "Device Categories to check: " + ', '.join(devices_list) + "\n\n"
        
        "Policy Text to Analyze:\n"
        + policy_text +
        "\n\n"
        
        "Your response should only be:\n"
        "- The device category that is mentioned in the policy text.\n"
        "- False: if none of the device categories are mentioned in the policy text.\n\n"
        
        "STRICT OUTPUT REQUIREMENT: Do not include any additional information, text, or explanation. Your response must "
        "be a single device category from the list or 'False'."
    )
    
    try:
        # Use Ollama library to interact with the model
        response = ollama.generate(model=model, prompt=prompt)
        return response # Ensure response is properly stripped to avoid whitespace issues
    except Exception as e:
        print(f"Error: {e}")
        return None
def check_device_in_policy_with_ollama3(policy_text, model="llama3.1"):
    # Define the prompt for analyzing the policy text
    prompt = (
        "You are a precise and logical text analysis assistant. Analyze the following company policy text and determine if "
        "it **clearly and explicitly mentions and references any smart devices** anywhere in the text. "
        "Return 'Device: ' followed by the smart device that is mentioned, or 'Device: None' if no such device is mentioned. "
        "Provide no additional output, explanation, or text other than 'Device: ' followed by the smart device mentioned or 'None'.\n\n"
        
        "Policy Text to Analyze:\n"
        + policy_text +
        "\n\n"
        
        "Your response should only be:\n"
        "- 'Device: ' followed by the smart device that is mentioned in the policy text.\n"
        "- 'Device: None' if no smart devices are mentioned in the policy text.\n\n"
        
        "STRICT OUTPUT REQUIREMENT: Do not include any additional information, text, or explanation. Your response must "
        "start with 'Device: ' followed by the smart device mentioned or 'None'."
    )

    try:
        # Use Ollama library to interact with the model
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Ensure response is properly stripped to avoid whitespace issues
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_device_in_policy_with_ollama4(policy_text, model="llama3.1"):
    # Define the prompt for analyzing the policy text
    prompt = (
        "You are a precise and logical text analysis assistant. Analyze the following company policy text and determine if "
        "it **clearly and explicitly mentions the company's own smart device and whether the policy applies to the smart device itself** anywhere in the text. "
        "Return 'Device: ' followed by the smart device if it is mentioned and covered by the policy, or 'Device: None' if no such device is mentioned or covered. "
        "Provide no additional output, explanation, or text other than 'Device: ' followed by the smart device or 'None'.\n\n"
        
        "Policy Text to Analyze:\n"
        + policy_text +
        "\n\n"
        
        "Your response should only be:\n"
        "- 'Device: ' followed by the company's smart device if it is mentioned and covered by the policy text.\n"
        "- 'Device: None' if the company's smart device is not mentioned or not covered by the policy text.\n\n"
        
        "STRICT OUTPUT REQUIREMENT: Do not include any additional information, text, or explanation. Your response must "
        "start with 'Device: ' followed by the company's smart device or 'None'."
    )

    try:
        # Use Ollama library to interact with the model
        response = ollama.generate(model=model, prompt=prompt)
        return response  # Ensure response is properly stripped to avoid whitespace issues
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_response3(response):
    try:
        # Ensure `response` is a string
        if isinstance(response, dict):
            response_text = response.get("response", "")
        else:
            response_text = response

        response_text = response_text.strip()

        # Check if the response starts with 'Device:'
        if response_text.lower().startswith('device:'):
            # Extract the text after 'Device:'
            device_info = response_text[len('Device:'):].strip()
            # Only take the first line to avoid capturing any additional text
            device_info = device_info.split('\n')[0].strip()
            if device_info.lower() == 'none':
                return None  # Return None if 'Device: None' is received
            else:
                return device_info  # Return only the device mentioned
        else:
            return None  # The response does not match the expected format
    except Exception as e:
        print(f"Error: {e}")
        return None


def parse_response(response):
    try:
        # Ensure `response` is the expected string, not a dictionary
        if isinstance(response, dict):
            response_text = response.get("response")  # Extract the string part of the response
        else:
            response_text = response

        # Use a regular expression to extract the similarity score
        match = re.search(r"True|False", response_text)
        if match:
            return match.group()
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def parse_response_full(response, devices_list):
    try:
        # Ensure `response` is a string
        if isinstance(response, dict):
            response_text = response.get("response", "")
        else:
            response_text = response

        response_text = response_text.lower()

        # Check if 'false' is in the response text
        if 'false' in response_text:
            return False

        # Create a regex pattern to match any device category
        devices_pattern = r'\b(' + '|'.join(re.escape(device.lower()) for device in devices_list) + r')\b'

        # Search for any device category in the response
        matches = re.findall(devices_pattern, response_text)
        if matches:
            # Return the first matched device category with original casing
            for device in devices_list:
                if device.lower() == matches[0]:
                    return device
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    
def check_all_devices_in_policy(df, devices_list):
    results = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        manufacturer = row['manufacturer']
        policy_text = row['policy_text']
        response = check_device_in_policy_with_ollama4(policy_text)
     
        response = response.get('response')
        response = parse_response3(response)
        devicecheck = False
        if response is False or response is None or response == "None":
            response = []
        if len(response) > 0:
            devicecheck = True
        results.append({"manufacturer": manufacturer, "devices": response, "DeviceCheck": devicecheck, "policy_text": policy_text})
    return results
def check_device_in_policies(df, device_list):
    results = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        manufacturer = row['manufacturer']
        policy_text = row['policy_text']
        found_devices = []
        for device in device_list:
            response = check_device_in_policy_with_ollama(policy_text, device)
            response = parse_response(response.get('response'))
            if response is True:
                print(f"Device: {device} mentioned in the policy text")
                found_devices.append(device)
        if len(found_devices) > 0:
            check = True
        else:
            check = False
        results.append({"manufacturer": manufacturer, "devices": found_devices, "policy_text": policy_text, "device_mentioned": check})
    return results

def device_count(df):
    count = 0
    for index, row in df.iterrows():
        if row['DeviceCheck'] == True:
            count += 1
    print(f"Number of policies mentioning a device: {count}")
    
    return count

def combine_df(df1, df2):
    df = pd.concat([df1, df2])
    return df

if __name__ == "__main__":
    """
    df = load_results('final_data.json')
    df = check_all_devices_in_policy(df, device_list)
    save_results(df, 'device_check.json')
    
    df1 = load_results('google_play_wayback.json')
    df1 = check_all_devices_in_policy(df1, device_list)
    save_results(df1, 'devicecheck_wayback.json')
    """
    df = load_results('device_check.json')
    count = device_count(df)
    print(f"Number of policies mentioning a device: {count}")
    print(f"Total number of policies: {len(df)}")