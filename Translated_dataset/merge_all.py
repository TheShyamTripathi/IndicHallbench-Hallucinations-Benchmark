
import json
from pathlib import Path
from collections import Counter

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(r"E:\Neural Project")

QUESTION_FILE = (
    BASE_DIR / "Translated_dataset" / "all_dataset_clean.json"
)

RESPONSE_FILE = (
    BASE_DIR / "LLM_Resoponse" / "model_responses_clean.json"
)

OUTPUT_FILE = BASE_DIR / "final_merged_dataset.json"
REPORT_FILE = BASE_DIR / "merge_validation_report.json"


# ============================================================
# LOAD JSON FILES
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


questions = load_json(QUESTION_FILE)
responses = load_json(RESPONSE_FILE)

if not isinstance(questions, list):
    raise ValueError("Question dataset must be a JSON list.")

if not isinstance(responses, list):
    raise ValueError("Model responses must be a JSON list.")

print(f"Questions loaded: {len(questions)}")
print(f"Responses loaded: {len(responses)}")


# ============================================================
# INDEX CANONICAL QUESTIONS
# ============================================================

question_index = {}

for question in questions:
    qid = str(question.get("question_id", "")).strip()

    if not qid:
        raise ValueError("Question without question_id found.")

    if qid in question_index:
        raise ValueError(f"Duplicate question ID in dataset: {qid}")

    question_index[qid] = question


# ============================================================
# MERGE RESPONSES
# One output row per model response.
# Repeated question IDs are allowed.
# ============================================================

output_columns = [
    "question_id",
    "fact_id",
    "domain",
    "difficulty",
    "fact_type",
    "language",
    "model_name",
    "response",
    "timestamp",
    "source_reference",
]

merged_records = []
unknown_question_ids = []
missing_fields = []
duplicate_response_keys = []
seen_response_keys = set()

for response in responses:

    qid = str(response.get("question_id", "")).strip()
    model_name = str(response.get("model_name", "")).strip()

    if not qid or not model_name:
        missing_fields.append({
            "question_id": qid,
            "model_name": model_name,
            "issue": "Missing question_id or model_name"
        })
        continue

    # A response is identified by question + model.
    # Model version is intentionally excluded.
    response_key = (qid, model_name)

    if response_key in seen_response_keys:
        duplicate_response_keys.append({
            "question_id": qid,
            "model_name": model_name,
            "issue": "Repeated question/model combination"
        })
        continue

    seen_response_keys.add(response_key)

    if qid not in question_index:
        unknown_question_ids.append({
            "question_id": qid,
            "model_name": model_name
        })
        continue

    question = question_index[qid]

    # Select the required fields only.
    record = {
        "question_id": qid,
        "fact_id": question.get("fact_id"),
        "domain": question.get("domain"),
        "difficulty": question.get("difficulty"),

        # Prefer fact_type from the dataset; fall back to
        # question_type if that is how your dataset stores it.
        "fact_type": (
            question.get("fact_type")
            or question.get("question_type")
        ),

        # Prefer response language, then question language.
        "language": (
            response.get("language")
            or question.get("language")
        ),

        "model_name": model_name,
        "response": response.get("response"),
        "timestamp": response.get("timestamp"),

        # Prefer the canonical source URL.
        "source_reference": (
            question.get("source_reference")
            or question.get("source_url")
            or question.get("source_reference_url")
        ),
    }

    # Report missing required response values.
    for field in ["response", "timestamp"]:
        if record[field] is None or record[field] == "":
            missing_fields.append({
                "question_id": qid,
                "model_name": model_name,
                "issue": f"Missing response field: {field}"
            })

    merged_records.append(record)


# ============================================================
# VALIDATION REPORT
# ============================================================

model_counts = Counter(
    row["model_name"] for row in merged_records
)

question_counts = Counter(
    row["question_id"] for row in merged_records
)

report = {
    "canonical_question_count": len(questions),
    "input_response_count": len(responses),
    "merged_record_count": len(merged_records),
    "model_counts": dict(model_counts),
    "questions_with_response_counts": dict(
        Counter(question_counts.values())
    ),
    "unknown_question_ids": unknown_question_ids,
    "duplicate_response_keys": duplicate_response_keys,
    "missing_fields": missing_fields,
}


# ============================================================
# SAVE OUTPUT
# ============================================================

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        merged_records,
        file,
        ensure_ascii=False,
        indent=2
    )

with REPORT_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========== MERGE SUMMARY ==========")
print(f"Canonical questions: {len(questions)}")
print(f"Input responses: {len(responses)}")
print(f"Merged records: {len(merged_records)}")

print("\nRecords per model:")
for model, count in sorted(model_counts.items()):
    print(f"  {model}: {count}")

print("\nQuestions grouped by response count:")
for count, number_of_questions in sorted(
    report["questions_with_response_counts"].items()
):
    print(f"  {count} response(s): {number_of_questions} question(s)")

print(f"\nUnknown question IDs: {len(unknown_question_ids)}")
print(f"Duplicate question/model keys: {len(duplicate_response_keys)}")
print(f"Missing-field issues: {len(missing_fields)}")

print(f"\nMerged dataset: {OUTPUT_FILE}")
print(f"Validation report: {REPORT_FILE}")
