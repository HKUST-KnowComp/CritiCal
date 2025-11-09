import json

data = []
for t in ["simple", "hard"]:
    with open(f"./math_perturb_{t}.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                item["type"] = t
                data.append(item)

with open("./math_perturb.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2) 

print(len(data))