#!/usr/bin/env python3
import argparse
import json
from datetime import date
from pathlib import Path

CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"


def resolve_tags(text: str, context: dict) -> str:
    key = context.get("key", "")
    value = context.get("value", "")
    result = context.get("result")
    if result is None:
        result = f"{key}={value}" if key and value else ""
    dt = context.get("date", date.today().isoformat())
    resolved = text.replace("$date", dt)
    resolved = resolved.replace("$key", key)
    resolved = resolved.replace("$value", value)
    resolved = resolved.replace("$result", result)
    return resolved


def format_chatml(messages, context: dict) -> str:
    parts = []
    for msg in messages:
        role = msg["role"].strip().lower()
        raw_content = msg["content"]
        content = resolve_tags(raw_content, context).strip() if role == "assistant" else raw_content.strip()
        parts.append(f"{CHATML_START}{role}\n{content}\n{CHATML_END}")
    return "\n".join(parts)


def convert_record(record: dict) -> str:
    if "messages" in record:
        return format_chatml(record["messages"], record)
    prompt = record.get("prompt", "")
    response = resolve_tags(record.get("response", ""), record)
    return (
        f"{CHATML_START}system\nYou are DanexChat, a helpful assistant.\n{CHATML_END}\n"
        f"{CHATML_START}user\n{prompt}\n{CHATML_END}\n"
        f"{CHATML_START}assistant\n{response}\n{CHATML_END}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chat-style JSONL into ChatML text for SFT")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lines = []
    with args.input.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            lines.append(convert_record(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lines)} training rows to {args.output}")


if __name__ == "__main__":
    main()
