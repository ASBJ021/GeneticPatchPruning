#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def expand_inputs(inputs):
    files = []
    for item in inputs:
        matched = sorted(glob.glob(item))
        if matched:
            files.extend(matched)
        else:
            files.append(item)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Merge JSONL files, de-duplicate by key, and sort."
    )
    parser.add_argument(
        "inputs", nargs="+", help="Input JSONL files or glob patterns"
    )
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--dedup-key",
        default="image_id",
        help="Key to de-duplicate by (default: image_id)",
    )
    parser.add_argument(
        "--keep",
        choices=("first", "last"),
        default="first",
        help="Which record to keep when duplicates appear (default: first)",
    )
    args = parser.parse_args()

    files = expand_inputs(args.inputs)
    records = {}
    order = []

    for path in files:
        if not os.path.exists(path):
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        for line in iter_jsonl(path):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"skip bad json in {path}", file=sys.stderr)
                continue
            key = obj.get(args.dedup_key)
            if key is None:
                print(f"skip missing {args.dedup_key} in {path}", file=sys.stderr)
                continue
            if key not in records:
                order.append(key)
            if args.keep == "first" and key in records:
                continue
            records[key] = obj

    sorted_keys = sorted(records.keys())
    with open(args.output, "w", encoding="utf-8") as out:
        for key in sorted_keys:
            out.write(json.dumps(records[key], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
