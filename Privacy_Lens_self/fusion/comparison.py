import os
import pandas as pd
import json
import re

# -----------------------------
# Utility functions for I/O
# -----------------------------
def load_results(file_path):
    """Load a JSON results file into a DataFrame."""
    return pd.read_json(file_path)

def save_results(results, file_path):
    """Save a DataFrame (or list of dicts) to a JSON file."""
    pd.DataFrame(results).to_json(file_path, orient="records", indent=4)

# -----------------------------
# Summary Functions for each test
# -----------------------------

def summary_ambiguity(df):
    # Assumes df has a column "ambiguity_level" with values 1,2,3.
    count_1 = (df["ambiguity_level"] == 1).sum()
    count_2 = (df["ambiguity_level"] == 2).sum()
    count_3 = (df["ambiguity_level"] == 3).sum()
    print("=== Ambiguity Summary ===")
    print(f"Total Policies: {len(df)}")
    print(f"Ambiguity Level 1: {count_1}")
    print(f"Ambiguity Level 2: {count_2}")
    print(f"Ambiguity Level 3: {count_3}")

def summary_devicecheck(df):
    # Assumes df has a boolean column "DeviceCheck".
    true_count = (df["DeviceCheck"] != False).sum()
    false_count = (df["DeviceCheck"] == False).sum()
    print("=== DeviceCheck Summary ===")
    print(f"Total Policies: {len(df)}")
    print(f"DeviceCheck True: {true_count}")
    print(f"DeviceCheck False: {false_count}")

def summary_keyword(df):
    # Columns that are not keyword counts.
    non_key = {"manufacturer", "policy_text", 'id', 'category', 'website', 'multilingual', 'recheck'}
    print("=== Keyword Summary ===")
    for col in df.columns:
        if col not in non_key:
            # Filter out keyword counts above 100.
            filtered = df[df[col] <= 1000000][col]
            print(len(filtered))
            if filtered.empty:
                print(f"Category '{col}': No data (all values > 100).")
            else:
                maximum = filtered.max()
                minimum = filtered.min()
                avg = filtered.median()
                print(f"Category '{col}': Max = {maximum}, Min = {minimum}, Average = {avg:.2f}")


def summary_readability(df):
    # Expected numeric columns for readability.
    numeric_cols = ["coherence_score", "imprecise_words_frequency", "connective_words_frequency",
                    "reading_complexity", "reading_time", "entropy", "unique_words_frequency", "grammatical_errors"]
    available = [col for col in numeric_cols if col in df.columns]
    print("=== Readability Summary ===")
    if available:
        print(df[available].describe())
    else:
        print("No readability metrics found.")

def summary_update_year(df):
    # Assumes df has "last_updated_year"
    print("=== Update Year Summary ===")
    counts = df["last_updated_year"].value_counts().sort_index()
    for year, cnt in counts.items():
        print(f"Year {year}: {cnt} policies")
    print(f"Total Policies: {len(df)}")

def flatten_similarity(sim_results):
    """
    The similarity results are structured as a list of dictionaries,
    each with "name" (manufacturer) and "series" (a list of comparisons with keys "name" and "similarity").
    This function flattens that into a DataFrame.
    """
    rows = []
    for item in sim_results:
        policy_name = item.get("name")
        for comp in item.get("series", []):
            rows.append({
                "policy_name": policy_name,
                "compared_to": comp.get("name"),
                "similarity": comp.get("similarity")
            })
    return pd.DataFrame(rows)

def summary_similarity(sim_df):
    # sim_df is a flattened DataFrame with a "similarity" column.
    print("=== Similarity Summary ===")
    if not sim_df.empty:
        print(sim_df["similarity"].describe())
    else:
        print("No similarity data available.")

# -----------------------------
# Splitting Utility Function
# -----------------------------
def split_by_devicecheck(test_df, device_df):
    """
    Merge test_df with device_df on "manufacturer" and "policy_text" to get the DeviceCheck flag,
    then return two DataFrames: one for DeviceCheck == True and one for DeviceCheck == False.
    """
    merged = pd.merge(test_df, device_df[["manufacturer", "policy_text", "DeviceCheck"]],
                      on=["manufacturer", "policy_text"], how="left")
    return merged[merged["DeviceCheck"] == True], merged[merged["DeviceCheck"] == False]

# -----------------------------
# Master Summary Functions for each test
# (Full summary and split summary)
# -----------------------------

def full_summary(test_name, df, summary_fn):
    print(f"\n========== Full Summary for {test_name} ==========")
    summary_fn(df)
    print("="*50)

def split_summary(test_name, test_df, device_df, summary_fn):
    print(f"\n========== Split Summary for {test_name} (by DeviceCheck) ==========")
    true_df, false_df = split_by_devicecheck(test_df, device_df)
    print(f"\n--- {test_name} for DeviceCheck = True (n={len(true_df)}) ---")
    summary_fn(true_df)
    print(f"\n--- {test_name} for DeviceCheck = False (n={len(false_df)}) ---")
    summary_fn(false_df)
    print("="*50)

# -----------------------------
# Main function that loads each test's result and outputs summaries
# -----------------------------
def main():
    # Define file names for each test's final results.
    files = {
        "Ambiguity": "ambiguity_results.json",
        "DeviceCheck": "devicecheck_results.json",
        "Keyword": "keyword_results.json",
        "Readability": "readability_final_results.json",
        "UpdateYear": "last_update_final_results.json",
        "Similarity": "similarity_final_results.json"
    }
    
    results = {}
    for test, fname in files.items():
        if os.path.exists(fname):
            results[test] = load_results(fname)
        else:
            print(f"File {fname} for test {test} not found.")
            results[test] = None

    # First, print full summaries for each test.
    if results["Ambiguity"] is not None:
        full_summary("Ambiguity", results["Ambiguity"], summary_ambiguity)
    if results["DeviceCheck"] is not None:
        full_summary("DeviceCheck", results["DeviceCheck"], summary_devicecheck)
    if results["Keyword"] is not None:
        full_summary("Keyword", results["Keyword"], summary_keyword)
    if results["Readability"] is not None:
        full_summary("Readability", results["Readability"], summary_readability)
    if results["UpdateYear"] is not None:
        full_summary("Update Year", results["UpdateYear"], summary_update_year)
    if results["Similarity"] is not None:
        # Flatten the similarity results
        sim_flat = flatten_similarity(results["Similarity"].to_dict(orient="records"))
        full_summary("Similarity", sim_flat, summary_similarity)
    
    # Now, for tests that should be split based on DeviceCheck, use the DeviceCheck results.
    device_df = results["DeviceCheck"]
    if device_df is not None:
        if results["Ambiguity"] is not None:
            split_summary("Ambiguity", results["Ambiguity"], device_df, summary_ambiguity)
        if results["Keyword"] is not None:
            split_summary("Keyword", results["Keyword"], device_df, summary_keyword)
        if results["Readability"] is not None:
            split_summary("Readability", results["Readability"], device_df, summary_readability)
        if results["UpdateYear"] is not None:
            split_summary("Update Year", results["UpdateYear"], device_df, summary_update_year)
        # Similarity splitting can be tricky because of its nested structure.
        # For simplicity, we skip splitting similarity.
        print("\nSimilarity split summary not provided due to nested structure complexities.")
    else:
        print("DeviceCheck results not available; cannot perform split summaries.")

if __name__ == "__main__":
    main()
