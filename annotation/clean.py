import json
from pathlib import Path

input_file = "subset-ranked.jsonl"
output_file = "subset-ranked-cleaned.jsonl"

def load_concatenated_json(filepath):
    text = Path(filepath).read_text(encoding="utf-8")

    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)

    while idx < length:
        # Skip whitespace/newlines
        while idx < length and text[idx].isspace():
            idx += 1

        if idx >= length:
            break

        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end


# Convert to JSONL
with open(output_file, "w", encoding="utf-8") as fout:
    for obj in load_concatenated_json(input_file):
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Saved JSONL to: {output_file}")
