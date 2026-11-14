import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness_label": {
            "type": "string",
            "enum": ["Correct", "Partially Correct", "Incorrect", "Uncertain"]
        },
        "hallucination_types": {
            "type": "array",
            "items": {"type": "string", "enum": [
                "None", "Entity Error", "Numerical Error", "Temporal Error",
                "Location Error", "Fabrication", "Contradiction", "Other"
            ]}
        },
        "severity": {
            "type": "string",
            "enum": ["None", "Minor", "Moderate", "Critical"]
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "assessment": {
                        "type": "string",
                        "enum": ["Supported", "Contradicted", "NotEstablished", "Unclear"]
                    },
                    "hallucination_type": {
                        "type": "string",
                        "enum": ["None", "Entity Error", "Numerical Error", "Temporal Error",
                                 "Location Error", "Fabrication", "Contradiction", "Other"]
                    },
                    "explanation": {"type": "string"}
                },
                "required": ["claim", "assessment", "hallucination_type", "explanation"],
                "additionalProperties": False
            }
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "review_required": {"type": "boolean"}
    },
    "required": [
        "correctness_label", "hallucination_types", "severity",
        "claims", "reason", "confidence", "review_required"
    ],
    "additionalProperties": False
}


def load_json_array(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def stable_response_id(row):
    # Timestamp differentiates repeated model runs; response hash protects against
    # accidental collisions when timestamps are missing or duplicated.
    raw = "|".join(str(row.get(k, "")) for k in
                   ("question_id", "language", "model_name", "timestamp", "response"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def build_reference_index(references):
    index = {}
    duplicates = []
    for i, ref in enumerate(references):
        qid = str(ref.get("question_id", "")).strip()
        if not qid:
            raise ValueError(f"Reference row {i} has no question_id.")
        if qid in index:
            duplicates.append(qid)
        else:
            index[qid] = ref
    if duplicates:
        examples = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(
            "Duplicate question_id values found in references.json. "
            f"Resolve them before annotation. Examples: {examples}"
        )
    return index


def make_messages(ref, response_row, template):
    # Do not invent evidence: this schema has ground_truth and source_url,
    # but no extracted source passage.
    payload = {
        "question_id": ref.get("question_id"),
        "fact_id": ref.get("fact_id", response_row.get("fact_id")),
        "domain": ref.get("domain", response_row.get("domain")),
        "difficulty": ref.get("difficulty", response_row.get("difficulty")),
        "question_type": ref.get("question_type", response_row.get("fact_type")),
        "question": ref.get("question"),
        "ground_truth": ref.get("ground_truth"),
        "answer_type": ref.get("answer_type"),
        "answer_length_category": ref.get("answer_length_category"),
        "primary_hallucination_type": ref.get("primary_hallucination_type"),
        "source_url": ref.get("source_url", response_row.get("source_url")),
        "language": response_row.get("language"),
        "model_name": response_row.get("model_name"),
        "response": response_row.get("response")
    }
    user_content = (
        "Annotate this single model response using the rules in your system prompt.\n"
        "Remember: source_url is provenance only; webpage contents are not supplied.\n\n"
        "RECORD:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": template},
        {"role": "user", "content": user_content}
    ]


def call_ollama(base_url, model, messages, timeout, num_ctx, num_predict):
    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": ANNOTATION_SCHEMA,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict
        }
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Empty response from Ollama: {data}")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ollama returned invalid JSON: {content[:1000]}") from e
    return result, data.get("prompt_eval_count"), data.get("eval_count")


def validate_result(result):
    required = {
        "correctness_label", "hallucination_types", "severity",
        "claims", "reason", "confidence", "review_required"
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"Missing annotation fields: {sorted(missing)}")
    if result["correctness_label"] not in {
        "Correct", "Partially Correct", "Incorrect", "Uncertain"
    }:
        raise ValueError("Invalid correctness_label.")
    if result["severity"] not in {"None", "Minor", "Moderate", "Critical"}:
        raise ValueError("Invalid severity.")
    if not isinstance(result["confidence"], (int, float)) or not 0 <= result["confidence"] <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if not isinstance(result["claims"], list):
        raise ValueError("claims must be a list.")
    for claim in result["claims"]:
        if claim.get("assessment") not in {"Supported", "Contradicted", "NotEstablished", "Unclear"}:
            raise ValueError("Invalid claim assessment.")
    # Conservative guardrails: unresolved claims must be reviewed.
    if any(c.get("assessment") in {"NotEstablished", "Unclear"} for c in result["claims"]):
        result["review_required"] = True
    if result["correctness_label"] == "Uncertain":
        result["review_required"] = True
    return result


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("response_id"):
                    completed.add(obj["response_id"])
            except json.JSONDecodeError:
                print("WARNING: Skipping malformed line in existing annotations file.")
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Annotate responses.json against references.json using local Ollama."
    )
    parser.add_argument("--responses", default="input/responses.json")
    parser.add_argument("--references", default="input/references.json")
    parser.add_argument("--prompt", default="prompts/annotation_prompt.txt")
    parser.add_argument("--output", default="output/annotations.jsonl")
    parser.add_argument("--errors", default="output/annotation_errors.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0,
                        help="Annotate only first N pending rows; 0 means all.")
    parser.add_argument("--retry", type=int, default=3)
    args = parser.parse_args()

    output_path = Path(args.output)
    error_path = Path(args.errors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    responses = load_json_array(args.responses)
    references = load_json_array(args.references)
    ref_index = build_reference_index(references)
    template = Path(args.prompt).read_text(encoding="utf-8")

    completed = read_completed(output_path)
    pending = []
    missing_refs = []
    for row_number, row in enumerate(responses, start=1):
        qid = str(row.get("question_id", "")).strip()
        if not qid:
            missing_refs.append({"row_number": row_number, "error": "Missing question_id"})
            continue
        rid = stable_response_id(row)
        if rid in completed:
            continue
        ref = ref_index.get(qid)
        if ref is None:
            missing_refs.append({
                "row_number": row_number, "question_id": qid,
                "error": "No matching question_id in references.json"
            })
            continue
        # Check key fields so a mismatched file is not silently annotated.
        if row.get("fact_id") and ref.get("fact_id") and row["fact_id"] != ref["fact_id"]:
            missing_refs.append({
                "row_number": row_number, "question_id": qid,
                "error": f"fact_id mismatch: response={row['fact_id']} reference={ref['fact_id']}"
            })
            continue
        if not str(row.get("response", "")).strip():
            missing_refs.append({
                "row_number": row_number, "question_id": qid,
                "error": "Empty response"
            })
            continue
        pending.append((row_number, rid, row, ref))

    for issue in missing_refs:
        append_jsonl(error_path, issue)

    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"Responses loaded: {len(responses)}")
    print(f"References loaded: {len(references)}")
    print(f"Already annotated: {len(completed)}")
    print(f"Pending this run: {len(pending)}")
    print(f"Rows skipped for input/reference problems: {len(missing_refs)}")
    print(f"Ollama model: {args.model}")

    success = 0
    failed = 0
    for row_number, rid, row, ref in tqdm(pending, desc="Annotating"):
        messages = make_messages(ref, row, template)
        last_error = None
        for attempt in range(1, args.retry + 1):
            try:
                result, prompt_tokens, output_tokens = call_ollama(
                    args.base_url, args.model, messages,
                    args.timeout, args.num_ctx, args.num_predict
                )
                result = validate_result(result)
                record = {
                    "response_id": rid,
                    "question_id": row.get("question_id"),
                    "fact_id": row.get("fact_id"),
                    "language": row.get("language"),
                    "model_name": row.get("model_name"),
                    "timestamp": row.get("timestamp"),
                    "judge_model": args.model,
                    "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
                    **result,
                    "judge_prompt_tokens": prompt_tokens,
                    "judge_output_tokens": output_tokens
                }
                append_jsonl(output_path, record)
                success += 1
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                if attempt < args.retry:
                    time.sleep(min(5 * attempt, 20))

        if last_error is not None:
            failed += 1
            append_jsonl(error_path, {
                "row_number": row_number,
                "response_id": rid,
                "question_id": row.get("question_id"),
                "fact_id": row.get("fact_id"),
                "language": row.get("language"),
                "model_name": row.get("model_name"),
                "error": last_error,
                "attempts": args.retry
            })

    print("\nRun complete.")
    print(f"Successfully annotated: {success}")
    print(f"Failed after retries: {failed}")
    print(f"Annotations: {output_path.resolve()}")
    print(f"Errors: {error_path.resolve()}")
    print("You can rerun the same command: completed response_ids will be skipped.")


if __name__ == "__main__":
    main()
