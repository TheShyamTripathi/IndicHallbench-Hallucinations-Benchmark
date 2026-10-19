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
# Repeated question IDs are EXPECTED and allowed.
# ============================================================

# Exact 10 columns requested
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
    "source_url",
]

merged_records = []
unknown_question_ids = []
missing_fields = []

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

    if qid not in question_index:
        unknown_question_ids.append({
            "question_id": qid,
            "model_name": model_name
        })
        continue

    question = question_index[qid]

    # Select the required fields and map them to exact column names
    record = {
        "question_id": qid,
        "fact_id": question.get("fact_id"),
        "domain": question.get("domain"),
        "difficulty": question.get("difficulty"),
        
        # Dataset uses "question_type" (e.g., "Entity"), fallback to "fact_type"
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

        # Renamed to source_url as requested
        "source_url": (
            question.get("source_url")
            or question.get("source_reference")
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
print(f"Missing-field issues: {len(missing_fields)}")

print(f"\nMerged dataset: {OUTPUT_FILE}")
print(f"Validation report: {REPORT_FILE}")