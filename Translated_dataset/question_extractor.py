import json
import os

# --- Configuration ---
DATASET_FILE = r"E:\Neural Project\Translated_dataset\dataset_en.json"   # Input JSON file
BATCH_SIZE = 180
TOTAL_BATCHES = 10
OUTPUT_DIR = r"E:\Neural Project\Translated_dataset\temp_batches2"


def load_dataset(filepath):
    """Load dataset from JSON file."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✅ Loaded {len(data)} records")
        return data

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def extract_batch(data, batch_number):
    """Extract a batch of records."""
    start_idx = (batch_number - 1) * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE

    if batch_number < 10:
        batch_data = data[start_idx:end_idx]
    else:
        batch_data = data[start_idx:]
    extracted = [
        {
            "question_id": record.get("question_id", ""),
            "question": record.get("question", "")
        }
        for record in batch_data
    ]

    return extracted


def save_batch(batch_data, batch_number):
    """Save batch to JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"temp_batch_{batch_number}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(batch_data)} records → {output_file}")


def main():
    data = load_dataset(DATASET_FILE)

    if data is None:
        return

    print(f"\nDataset Size: {len(data)}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Total Batches: {TOTAL_BATCHES}\n")

    while True:
        user_input = input(
            f"Enter batch number (1-{TOTAL_BATCHES}) or 'q' to quit: "
        ).strip()

        if user_input.lower() == "q":
            print("Exiting...")
            break

        try:
            batch_num = int(user_input)

            if not (1 <= batch_num <= TOTAL_BATCHES):
                print(f"⚠️ Enter a number between 1 and {TOTAL_BATCHES}")
                continue

            batch = extract_batch(data, batch_num)

            if not batch:
                print("❌ No records found for this batch.")
                continue

            save_batch(batch, batch_num)

            print("\nPreview:")
            for item in batch[:3]:
                print(item)

        except ValueError:
            print("⚠️ Please enter a valid number.")


if __name__ == "__main__":
    main()