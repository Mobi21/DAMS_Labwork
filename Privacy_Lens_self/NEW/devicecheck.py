# devicecheck.py
import json
import re
import pandas as pd
import ollama
import logging
from tqdm import tqdm

def load_results(filename="final_data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_results(results, filename):
    pd.DataFrame(results).to_json(filename, orient="records", indent=4)

def check_device_in_policy(policy_text, model="llama3.1"):
    prompt = """
You are an expert in legal and policy analysis. Analyze the provided company policy text and determine whether it explicitly mentions a smart device manufactured by the company.
Your response must be exactly in one of the two formats:
Device: <Smart Device Name>  or  Device: None

Policy text:
{policy_text}
"""
    try:
        response = ollama.generate(model=model, prompt=prompt.format(policy_text=policy_text))
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_device_response(response_text):
    response_text = response_text.strip()
    if response_text.lower().startswith("device:"):
        device_info = response_text[len("device:"):].strip().split("\n")[0].strip()
        return None if device_info.lower() == "none" else device_info
    return None

def check_all_devices(data, device_list):
    results = []
    logs = []
    for _, policy in tqdm(data.iterrows(), total=len(data), desc="DeviceCheck Analysis"):
        manufacturer = policy["manufacturer"]
        policy_text = policy["policy_text"]
        response_obj = check_device_in_policy(policy_text)
        log_entry = {"manufacturer": manufacturer, "policy_text": policy_text}
        if response_obj:
            log_entry.update(response_obj)
        logs.append(log_entry)
        response_text = response_obj.get("response") if response_obj and hasattr(response_obj, "get") else ""
        device_found = parse_device_response(response_text)
        results.append({
            "manufacturer": manufacturer,
            "devices": device_found if device_found else "",
            "DeviceCheck": True if device_found else False,
            "policy_text": policy_text
        })
    return results, logs

def run_tests(output_dir="results"):
    device_list = ["Smart Speaker", "Smart Thermostat", "Smart Camera", "Smart Lock", "Smart Fitness Tracker",
                   "Smart Light", "Smart Doorbell", "Smart Alarm System", "Smart TV", "Smart Scale",
                   "Smart Home Device", "Smart Air Purifier", "Smart Sensor", "Smart Watch", "Smart Monitor",
                   "Smart Security", "Smart Health Tracker", "Smart Refrigerator", "Smart Location Tracker",
                   "Smart Entertainment Device", "Smart Connected Vehicle", "Smart Networking",
                   "Smart Fitness Equipment", "Smart Mount", "Smart Projector", "Smart Body Scanners", "Smart Gaming"]
    df1 = load_results("final_data.json")
    results1, logs1 = check_all_devices(df1, device_list)
    df2 = load_results("google_play_wayback.json")
    results2, logs2 = check_all_devices(df2, device_list)
    final_results = results1 + results2
    final_logs = logs1 + logs2
    save_results(final_results, f"{output_dir}/devicecheck_results.json")
    save_results(final_logs, f"{output_dir}/devicecheck_logs.json")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_tests("results")
