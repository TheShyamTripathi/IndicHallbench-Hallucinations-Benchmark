import json
import glob

# 1. Find all json files starting with 'dataset_' in the current directory
# (Adjust the path if your files are in a subfolder, e.g., 'data/dataset_*.json')
file_list = glob.glob(r'E:\Neural Project\Translated_dataset\dataset_*.json')

merged_data = []

# 2. Loop through each file and add its contents to the master list
for file_name in file_list:
    print(f"Processing {file_name}...")
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f"Warning: {file_name} does not contain a list. Skipping.")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# 3. Save the combined data to a new file
output_filename = r'E:\Neural Project\Translated_dataset\all_dataset_clean.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print(f"Successfully merged {len(file_list)} files into {output_filename} with {len(merged_data)} total records.")