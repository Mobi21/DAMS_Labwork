import os
import importlib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_module_by_name(module_name, output_dir):
    """
    Imports the module by name, calls its run_tests(output_dir) function,
    and returns a tuple (results, logs). If only a DataFrame is returned, logs is None.
    """
    mod = importlib.import_module(module_name)
    output = mod.run_tests(output_dir=output_dir)
    if isinstance(output, tuple):
        return output  # (results, logs)
    else:
        return output, None

def summarize_df(df, test_name):
    print(f"=== {test_name} Summary ===")
    print(f"Total records: {len(df)}")
    print(df.head())
    print("=" * 40 + "\n")

def main():
    output_dir = "combined_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # List of module names (as strings) to run concurrently.
    modules = {
        "Ambiguity": "ambiguity",
        "DeviceCheck": "devicecheck",
        "UpdateYear": "update_year",
        "Readability": "readability",
        "Keyword": "keyword_extractor"
        # "Similarity": "similarity"  # Uncomment if desired.
    }
    
    final_results = {}
    final_logs = {}
    
    with ProcessPoolExecutor() as executor:
        future_to_name = {
            executor.submit(run_module_by_name, module_name, output_dir): name
            for name, module_name in modules.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results, logs = future.result()
                final_results[name] = results
                final_logs[name] = logs
                summarize_df(results, name)
                if logs is not None:
                    print(f"Logs for {name} (first 5 rows):")
                    print(logs.head())
                    print("=" * 40 + "\n")
            except Exception as exc:
                print(f"{name} generated an exception: {exc}")
    
    # Optionally, combine results if they share common columns.
    combined = []
    for name, df in final_results.items():
        if df is not None and "manufacturer" in df.columns and "policy_text" in df.columns:
            df = df.copy()
            df["test"] = name
            combined.append(df)
    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        combined_file = os.path.join(output_dir, "all_tests_combined_results.json")
        combined_df.to_json(combined_file, orient="records", indent=4)
        print("Combined results saved to:", combined_file)
    
    print("All tests completed. Results and logs have been saved in:", output_dir)

if __name__ == "__main__":
    main()
