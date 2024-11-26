import json
import re
import pandas as pd
import ollama  # Ensure Ollama is properly installed and configured
import time
import logging
from tqdm import tqdm
import random

def find_new(df1, df2):
    df2 = df2.rename(columns={"url": "manufacturer"})
    for index, row in df1.iterrows():
        if row["manufacturer"] in df2["manufacturer"].values:
            df1 = df1.drop(index)
            
    return df1

def load_results(filename="data.json"):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if filename == "results.json":
        df=df[['manufacturer', 'response']]

    return df



def combine_results(df1, df2):
    df1 = df1[["manufacturer", "policy_text"]]
    df2 = df2[["manufacturer", "policy_text"]]
    df = pd.concat([df1, df2])
    return df

def save_results(results, filename="results.json"):
    df = pd.DataFrame(results)
    df.to_json(filename, orient="records", indent=4)
if __name__ == "__main__":
    df1 = load_results("final_data.json")
    df2 = load_results("true_ambiguity.json")
    df3 = load_results("google_play_wayback.json")
    new_df = find_new(df1, df2)

    final_df = combine_results(new_df, df3)   
    print(len(final_df))
    print(final_df.columns)
    save_results(final_df, "detected.json")