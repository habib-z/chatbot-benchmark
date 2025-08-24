import json

import json
from pathlib import Path
p=Path("../prompts/faithfulness/statement_generator/v1/statement_generator.examples.jsonl")
out = p#.with_name(p.stem + ".compact.jsonl")

with p.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as o:
    buf = []
    for line in f:
        if line.strip() == "":
            if buf:
                obj = json.loads("".join(buf))
                o.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")
                buf = []
        else:
            buf.append(line)
    if buf:
        obj = json.loads("".join(buf))
        o.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

print("Wrote:", out)