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
    final_df = final_df.drop_duplicates(subset='manufacturer')
    print(f"Combined data has {len(final_df)} rows.")
    return final_df

if __name__ == "__main__":
    df1 = load_data("data.json")
    df2 = load_data("policies.json", True)
    final_df = combine_data(df1, df2)
    final_df.to_json("final_data.json", orient='records', indent = 4)

