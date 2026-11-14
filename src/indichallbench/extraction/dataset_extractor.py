import json
import os
import math

# --- Configuration ---
MERGED_DATASET_FILE = "dataset_merged_all.json"  # Your merged dataset file
BATCH_SIZE = 90
TOTAL_BATCHES = 20
TEMP_OUTPUT_DIR = "temp_batches"  # Folder to store temp batch files

def load_merged_dataset(filepath):
    """Loads the merged JSON dataset."""
    if not os.path.exists(filepath):
        print(f"❌ Error: Merged dataset file '{filepath}' not found.")
        print("Please ensure you have merged your dataset files first.")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded merged dataset with {len(data)} total records.")
        return data
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return None

def extract_batch(data, batch_number):
    """Extracts a specific batch of 90 records."""
    if batch_number < 1 or batch_number > TOTAL_BATCHES:
        print(f"❌ Error: Batch number must be between 1 and {TOTAL_BATCHES}.")
        return None
    
    # Calculate start and end indices (0-based indexing)
    start_idx = (batch_number - 1) * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    
    # Check if we have enough data
    if start_idx >= len(data):
        print(f"❌ Error: Dataset only has {len(data)} records. Batch {batch_number} starts at index {start_idx}.")
        return None
    
    # Slice the data
    batch_data = data[start_idx:end_idx]
    
    # Extract only the required fields
    extracted_batch = []
    for record in batch_data:
        extracted_batch.append({
            "question_id": record.get("question_id", "UNKNOWN_ID"),
            "question": record.get("question", "NO_QUESTION_FOUND"),
            "ground_truth": record.get("ground_truth", "NO_GROUND_TRUTH_FOUND")
        })
        
    return extracted_batch

def save_temp_batch(batch_data, batch_number):
    """Saves the extracted batch to a temporary JSON file."""
    if not os.path.exists(TEMP_OUTPUT_DIR):
        os.makedirs(TEMP_OUTPUT_DIR)
        
    filename = os.path.join(TEMP_OUTPUT_DIR, f"temp_batch_{batch_number}.json")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=4, ensure_ascii=False)
        print(f"✅ Successfully saved {len(batch_data)} records to '{filename}'")
        return filename
    except Exception as e:
        print(f"❌ Error saving temp file: {e}")
        return None

def main():
    print("="*50)
    print("📦 BATCH EXTRACTION TOOL")
    print("="*50)
    
    # 1. Load the merged dataset
    full_data = load_merged_dataset(MERGED_DATASET_FILE)
    if full_data is None:
        return
    
    # 2. Prompt user for batch number
    while True:
        try:
            user_input = input(f"\nEnter batch number (1 to {TOTAL_BATCHES}) or 'q' to quit: ").strip()
            
            if user_input.lower() == 'q':
                print("Exiting...")
                break
                
            batch_num = int(user_input)
            
            if batch_num < 1 or batch_num > TOTAL_BATCHES:
                print(f"⚠️ Invalid input. Please enter a number between 1 and {TOTAL_BATCHES}.")
                continue
                
            # 3. Extract the batch
            print(f"\nExtracting Batch {batch_num}...")
            batch = extract_batch(full_data, batch_num)
            
            if batch:
                # 4. Save to temp JSON file
                output_file = save_temp_batch(batch, batch_num)
                print(f"🎉 Batch {batch_num} is ready! You can now use '{output_file}' for LLM evaluation.")
                
                # Optional: Show first 3 records as preview
                print("\n--- Preview (First 3 records) ---")
                for i, record in enumerate(batch[:3]):
                    print(f"{i+1}. QID: {record['question_id']}")
                    print(f"   Question: {record['question'][:80]}...")
                    print(f"   Ground Truth: {record['ground_truth']}")
                print("-" * 30)
                
        except ValueError:
            print("⚠️ Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()