#%%

import pandas as pd
import json
import ast

def load_data(filename, drop_extra_columns=False):
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if drop_extra_columns:
        df = df[['manufacturer', 'policy_text']]
    return df

def  combine_data(df1, df2):
    final_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined data has {len(final_df)} rows.")
    return final_df

import pandas as pd

def split_dataset(dataset1, dataset2, device_check_col="DeviceCheck"):
    """
    Splits dataset2 into two subsets based on the values of the DeviceCheck column in dataset1.
    
    Parameters:
        dataset1 (pd.DataFrame): The first dataset containing the DeviceCheck column.
        dataset2 (pd.DataFrame): The second dataset to be split.
        device_check_col (str): The name of the column in dataset1 indicating true/false for splitting.
        
    Returns:
        pd.DataFrame: Subset of dataset2 where DeviceCheck is True.
        pd.DataFrame: Subset of dataset2 where DeviceCheck is False.
    """
    if device_check_col not in dataset1.columns:
        raise ValueError(f"Column '{device_check_col}' not found in dataset1.")

    if len(dataset1) != len(dataset2):
        raise ValueError("Datasets must have the same number of rows.")

    # Create masks for True and False
    true_mask = dataset1[device_check_col] == True
    false_mask = dataset1[device_check_col] == False

    # Split dataset2 using masks
    dataset2_true = dataset2[true_mask].reset_index(drop=True)
    dataset2_false = dataset2[false_mask].reset_index(drop=True)

    return dataset2_true, dataset2_false

if __name__ == "__main__":
    df1 = load_data("final_results.json")

    df2 = load_data("ambiguity_wayback.json", True)
    
    device_check = load_data("device_check.json")
    df2["manufacturer"] = df2["manufacturer"].astype(str) + "_wayback"
    
    final_df = combine_data(df1, df2)
    
    device_true, device_false = split_dataset(device_check, final_df)
    
    device_true.to_json("readability_true.json", orient="records", indent=4)
    device_false.to_json("readability_false.json", orient="records", indent=4)


# %%
