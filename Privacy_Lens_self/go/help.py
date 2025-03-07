import json

def combine_json_files(file1, file2, output_file):
    # Load the first JSON file
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    # Load the second JSON file
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # Combine the two lists
    combined = data1 + data2
    
    # Save the combined list to the output file
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(combined, out, indent=4)
    
    print(f"Combined {len(data1)} entries from {file1} and {len(data2)} entries from {file2} into {output_file}")

if __name__ == "__main__":
    # Change these file names as needed
    file1 = "keyword_results.json"
    file2 = "keyword_wayback.json"
    output_file = "device_check_results.json"
    
    combine_json_files(file1, file2, output_file)
