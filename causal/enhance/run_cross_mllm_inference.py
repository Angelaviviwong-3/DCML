#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a real alternative-MLLM scoring pass for the MCRE measurement audit.

The script reads the fixed audit item file created by
run_cross_mllm_measurement_audit.py, calls an OpenAI-compatible
vision-language model endpoint, and writes the filled alternative-score CSV
that can be evaluated by the audit script.
"""

from __future__ import annotations

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import argparse
import os
import base64
import csv
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


VERSION = "20260801"
ENHANCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENHANCE_DIR.parent.parent
OUTPUT_DIR = _dcml_paths.project_path("enhance/output")

CONSTRUCTS = [
    "T_con_mkt",
    "T_int_vis",
    "T_int_fac",
    "atomic_a_pri",
    "atomic_a_gft",
    "atomic_a_sub",
    "atomic_a_urg",
    "atomic_a_spec",
    "atomic_a_str",
]

CONSTRUCT_LABELS = {
    "T_con_mkt": "Marketing salience",
    "T_int_vis": "Functional visual salience",
    "T_int_fac": "Factual density",
    "atomic_a_pri": "Price shock",
    "atomic_a_gft": "Gift appeal",
    "atomic_a_sub": "Subsidy authenticity",
    "atomic_a_urg": "Urgency",
    "atomic_a_spec": "Specification overlay",
    "atomic_a_str": "Internal-structure view",
}

DEFAULT_INPUT = OUTPUT_DIR / f"Cross_MLLM_Audit_Input_Items_{VERSION}.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / f"Cross_MLLM_Audit_Alternative_Scores_{VERSION}.csv"
DEFAULT_LOG = OUTPUT_DIR / f"Cross_MLLM_Audit_Inference_Log_{VERSION}.jsonl"

RUBRIC = """
Score evidence strength, not whether the cue is expected to help sales.
Use the full 0-10 scale with these anchors: 0 absent; 1-3 weak evidence;
4-6 moderate evidence; 7-10 strong and salient evidence. Use 0 only when
both image and text provide no cue-specific evidence.

Construct definitions:
- T_con_mkt: overall promotional salience from price, gift, subsidy, or urgency cues.
- T_int_vis: overall functional visual salience from specification overlays or internal-structure/product-detail views.
- T_int_fac: density and diagnosticity of factual product text, including specifications, dimensions, materials, functions, compatibility, or technical claims.
- atomic_a_pri: explicit price, discount, markdown, coupon, or price-comparison cue.
- atomic_a_gft: explicit gift, bundle, bonus, accessory, or add-on cue.
- atomic_a_sub: explicit subsidy, reimbursement, platform allowance, or official promotion-credibility cue.
- atomic_a_urg: explicit urgency, limited-time, countdown, limited-stock, or scarcity cue.
- atomic_a_spec: visible or textual specification overlay, parameter list, model information, capacity, size, or technical labels.
- atomic_a_str: visible or described internal structure, cutaway, component view, mechanism, ingredient layout, or functional construction.
"""


def normalize_api_url(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    if api_base.endswith("/chat/completions"):
        return api_base
    return f"{api_base}/chat/completions"


def image_to_data_url(path_text: str) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_prompt(row: dict[str, str]) -> str:
    schema = "\n".join(f"- {code}: {label}" for code, label in CONSTRUCT_LABELS.items())
    text = (row.get("text_excerpt") or "")[:1800]
    return (
        "You are scoring multimodal e-commerce product content under a fixed MCRE rubric. "
        "Use the product image and product text together. If the image is plain but the text contains product attributes, "
        "score the relevant textual constructs from the text. Do not infer absent promotional cues from general product quality. "
        "The product text may use Chinese field labels: \u54c1\u724c means brand, \u7c7b\u76ee means category, \u6807\u9898 means title, "
        "\u5546\u54c1\u7279\u5f81 means product features, and \u8bc4\u8bba means reviews. "
        f"{RUBRIC}\n"
        "Return only valid JSON with this exact structure: "
        "{\"schema_valid\": true, \"scores\": {\"T_con_mkt\": 0, ...}, \"evidence\": {\"T_con_mkt\": \"brief evidence\", ...}}. "
        "The scores object must contain every construct code below.\n\n"
        f"Constructs:\n{schema}\n\n"
        f"Dataset: {row.get('dataset', '')}\n"
        f"Item ID: {row.get('item_id', '')}\n"
        f"Product text: {text}"
    )


def request_payload(
    row: dict[str, str],
    model_id: str,
    temperature: float,
    max_tokens: int,
    allow_missing_images: bool,
    disable_response_format: bool,
) -> dict:
    prompt = build_prompt(row)
    content: list[dict] = []
    data_url = image_to_data_url(row.get("image_path", ""))
    if data_url:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    elif not allow_missing_images:
        raise FileNotFoundError(f"Missing image for {row.get('dataset')} {row.get('item_id')}: {row.get('image_path')}")
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not disable_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def post_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:2000]}") from exc


def response_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_scores(text: str) -> tuple[bool, dict[str, float | None], str]:
    try:
        obj = parse_json_object(text)
    except Exception as exc:  # noqa: BLE001
        return False, {code: None for code in CONSTRUCTS}, f"json_parse_error: {exc}"
    raw_scores = obj.get("scores", obj.get("mcre_scores", obj))
    scores: dict[str, float | None] = {}
    ok = bool(obj.get("schema_valid", True))
    notes = []
    for code in CONSTRUCTS:
        value = raw_scores.get(code) if isinstance(raw_scores, dict) else None
        try:
            if value is None or value == "":
                scores[code] = None
                ok = False
                notes.append(f"missing_{code}")
            else:
                score = float(value)
                if score < 0 or score > 10:
                    notes.append(f"clipped_{code}")
                scores[code] = min(10.0, max(0.0, score))
        except Exception:  # noqa: BLE001
            scores[code] = None
            ok = False
            notes.append(f"non_numeric_{code}")
    return ok, scores, ";".join(notes)


def read_existing(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {(row["dataset"], row["item_id"]): row for row in reader}


def row_complete(row: dict[str, str]) -> bool:
    return all(str(row.get(f"{code}_0_10", "")).strip() != "" for code in CONSTRUCTS)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "item_id", "model_id", "schema_valid"]
    fieldnames += [f"{code}_0_10" for code in CONSTRUCTS]
    fieldnames += ["raw_response_or_file", "notes"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    url = normalize_api_url(args.api_base)
    existing = read_existing(args.output)
    completed = 0
    failed = 0
    rows_out: list[dict[str, str]] = []

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        input_rows = list(csv.DictReader(handle))

    if args.limit > 0:
        input_rows = input_rows[: args.limit]

    for idx, row in enumerate(input_rows, start=1):
        key = (row["dataset"], row["item_id"])
        if args.resume and key in existing and row_complete(existing[key]):
            rows_out.append(existing[key])
            completed += 1
            continue

        out_row = {
            "dataset": row["dataset"],
            "item_id": row["item_id"],
            "model_id": args.model_id,
            "schema_valid": "false",
            "raw_response_or_file": "",
            "notes": "",
        }
        for code in CONSTRUCTS:
            out_row[f"{code}_0_10"] = ""

        try:
            payload = request_payload(
                row=row,
                model_id=args.model_id,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                allow_missing_images=args.allow_missing_images,
                disable_response_format=args.disable_response_format,
            )
            response = None
            error = None
            for attempt in range(args.retries + 1):
                try:
                    response = post_json(url, payload, args.api_key, args.timeout)
                    error = None
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    error = exc
                    time.sleep(args.sleep * (attempt + 1))
            if error is not None:
                raise error
            content = response_text(response or {})
            schema_valid, scores, notes = parse_scores(content)
            out_row["schema_valid"] = "true" if schema_valid else "false"
            out_row["raw_response_or_file"] = content[:5000]
            out_row["notes"] = notes
            for code, value in scores.items():
                out_row[f"{code}_0_10"] = "" if value is None else f"{value:.4f}"
            if schema_valid:
                completed += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            out_row["notes"] = f"request_error: {exc}"
            failed += 1

        rows_out.append(out_row)
        append_log(
            args.log,
            {
                "index": idx,
                "total": len(input_rows),
                "dataset": row["dataset"],
                "item_id": row["item_id"],
                "schema_valid": out_row["schema_valid"],
                "notes": out_row["notes"],
            },
        )
        if idx % args.flush_every == 0:
            write_rows(args.output, rows_out)
            print(f"[{idx}/{len(input_rows)}] completed={completed} failed={failed}", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_rows(args.output, rows_out)
    print(f"Wrote alternative scores: {args.output}")
    print(f"Log file: {args.log}")
    print(f"completed={completed} failed={failed} total={len(input_rows)}")
    if failed:
        print("Some rows failed or were schema-invalid. Inspect the log and output CSV before evaluation.", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("DCML_MLLM_API_KEY", ""), help="API key if required; use a dummy value for local vLLM if needed.")
    parser.add_argument("--model-id", required=True, help="Alternative MLLM name served by the endpoint.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit. Use 0 for all rows.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send response_format=json_object. Use this for vLLM/model servers that reject response_format.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
