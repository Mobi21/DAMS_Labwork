# run_all_tests.py
import os
import ambiguity
import devicecheck
import keyword_extractor  # Avoid conflict with built-in 'keyword' module
import readability
import update_year
import similarity

def main():
    output_dir = "combined_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running Ambiguity Test...")
    ambiguity.run_tests(output_dir=output_dir)
    
    print("Running DeviceCheck Test...")
    devicecheck.run_tests(output_dir=output_dir)
    
    print("Running Keyword Extraction Test...")
    keyword_extractor.run_tests(output_dir=output_dir)
    
    print("Running Readability Test...")
    readability.run_tests(output_dir=output_dir)
    
    print("Running Update Year Test...")
    update_year.run_tests(output_dir=output_dir)
    
    print("Running Similarity Test...")
    similarity.run_tests(output_dir=output_dir)
    
    print("All tests completed. Results are saved in the folder:", output_dir)

if __name__ == "__main__":
    main()
