import json
import time
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI
import os
from prompt import *

# python gen_critique_all.py --task strategyqa --input_file ./data.json --mode conf --model_name gpt-4o --api_key YOUR_KEY --num_processes 10

# retry params
MAX_RETRIES = 3
RETRY_DELAY = 5

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, dir, name, task):
    if task == "comparisonqa":
        new_data = []
        for line in data[:900]:
            new_data.append({
                "instruction": line["high_prompt"],
                "input": line["high_response"],
                "output": line["high_critique_response"],
                "system": "You are a helpful assistant.",
            })
            new_data.append({
                "instruction": line["low_prompt"],
                "input": line["low_response"],
                "output": line["low_critique_response"],
                "system": "You are a helpful assistant.",
            })

        with open(f"{dir}/train_{name}", 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        print(f"training data len: {len(new_data)}")


        new_data = []
        for line in data[900:1000]:
            new_data.append({
                "instruction": line["high_prompt"],
                "input": line["high_response"],
                "output": line["high_critique_response"],
                "system": "You are a helpful assistant.",
            })
            new_data.append({
                "instruction": line["low_prompt"],
                "input": line["low_response"],
                "output": line["low_critique_response"],
                "system": "You are a helpful assistant.",
            })

        with open(f"{dir}/val_{name}", 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        print(f"validating data len: {len(new_data)}")

        with open(f"{dir}/{name}", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"all data len: {len(data)*2}")

    else:
        new_data = []
        for line in data[:1800]:
            new_data.append({
                "instruction": line["prompt"],
                "input": line["response"],
                "output": line["critique_response"],
                "system": "You are a helpful assistant.",
            })
        with open(f"{dir}/train_{name}", 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        print(f"training data len: {len(new_data)}")

        new_data = []
        for line in data[1800:2000]:
            new_data.append({
                "instruction": line["prompt"],
                "input": line["response"],
                "output": line["critique_response"],
                "system": "You are a helpful assistant.",
            })
        with open(f"{dir}/val_{name}", 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        print(f"validating data len: {len(new_data)}")

        with open(f"{dir}/{name}", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"all data len: {len(data)}")

def prepare_prompts(data, task, mode, think=True):
    """prepare critique prompt"""
    for item in data:
        if task == "comparisonqa":
            high_q = item.get("high_question", {})
            low_q = item.get("low_question", {})
            if mode == "unc":
                tmpl = GEN_CRIT_UNC_COMPARISONQA_THINK if think else GEN_CRIT_UNC_COMPARISONQA
            else:
                tmpl = GEN_CRIT_CONF_COMPARISONQA_THINK if think else GEN_CRIT_CONF_COMPARISONQA
            high_opts = high_q.get("options", {})
            low_opts = low_q.get("options", {})
            high_ans = list(high_q.get("answer", {}).keys())[0] if high_q.get("answer") else ""
            low_ans = list(low_q.get("answer", {}).keys())[0] if low_q.get("answer") else ""
            item["high_critique_prompt"] = tmpl.format(
                high_q.get("question", ""),
                high_opts.get("A", ""), high_opts.get("B", ""), high_opts.get("C", ""), high_opts.get("D", ""),
                high_ans,
                item.get("high_response", "")
            )
            item["low_critique_prompt"] = tmpl.format(
                low_q.get("question", ""),
                low_opts.get("A", ""), low_opts.get("B", ""), low_opts.get("C", ""), low_opts.get("D", ""),
                low_ans,
                item.get("low_response", "")
            )
            
        elif task == "math":
            question = item.get("problem")
            answer = item.get("solution")
            response = item.get("response")
            if mode == "unc":
                tmpl = GEN_CRIT_UNC_MATH_THINK if think else GEN_CRIT_UNC_MATH
            else:
                tmpl = GEN_CRIT_CONF_MATH_THINK if think else GEN_CRIT_CONF_MATH
            item["critique_prompt"] = tmpl.format(question, answer, response)

        elif task == "strategyqa":
            question = item.get("question")
            facts = item.get("facts")
            answer = item.get("answer")
            response = item.get("response")
        
            if mode == "unc":
                tmpl = GEN_CRIT_UNC_STRATEGYQA_THINK if think else GEN_CRIT_UNC_STRATEGYQA
            else:
                tmpl = GEN_CRIT_CONF_STRATEGYQA_THINK if think else GEN_CRIT_CONF_STRATEGYQA

            item["critique_prompt"] = tmpl.format(question, answer, facts, response)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
    return data

def call_api_with_retry(client, model_name, prompt, max_retries=MAX_RETRIES):
    last_err = None
    for i in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_body={},
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_err = e
            if i < max_retries - 1:
                time.sleep(RETRY_DELAY * (i + 1))
    raise last_err

def eval_item(args):
    """Evaluate a single item with retry mechanism."""
    line, task, model_name, api_key, base_url = args
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        if task == "comparisonqa":
            # high
            line['high_critique_response'] = call_api_with_retry(client, model_name, line['high_critique_prompt'])
            # low
            line['low_critique_response'] = call_api_with_retry(client, model_name, line['low_critique_prompt'])
        else:
            line['critique_response'] = call_api_with_retry(client, model_name, line['critique_prompt'])
        return line
    except Exception as e:
        print(f"Error")
        line['error'] = str(e)
        return line

def process_all(data, task, model_name, api_key, base_url, num_processes):
    args = [(line, task, model_name, api_key, base_url) for line in data]
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(eval_item, args), total=len(args)))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="strategyqa", choices=["strategyqa", "comparisonqa", "math"])
    parser.add_argument('--input_file', type=str, required=True, help='the response of the original model in training set')
    parser.add_argument('--model_name', type=str, default="openai/gpt-4o", help='the teacher model for critique generation')
    parser.add_argument('--origin_model', type=str, required=True, help='the original model name for constructing the name of output file')
    parser.add_argument('--mode', type=str, default="unc", choices=["unc", "conf"])
    parser.add_argument('--num_processes', type=int, default=50)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument('--think', action='store_true', help='use think separator (</think>)')
    args = parser.parse_args()

    data = load_json(args.input_file)
    if args.task == "comparisonqa":
        data = data[:1000]
    else:
        data = data[:2000]
    
    # prepare prompts
    data = prepare_prompts(data, args.task, args.mode, think=args.think)

    # process
    api_key = args.api_key
    if not api_key:
        raise SystemExit("api_key required")

    results = process_all(data, args.task, args.model_name, api_key, args.base_url, args.num_processes)

    # output path
    name = args.origin_model.split('/')[-1].replace('.', '').replace('-', '_').replace(':', '_')
    out_dir = f'../experiments/critique_train/{args.task}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'critique_{args.task}_{name}_{args.mode}_think.json' if args.think else f'critique_{args.task}_{name}_{args.mode}.json'
    save_json(results, out_dir, out_path, args.task)
    print(f"Saved to {out_dir}")

if __name__ == '__main__':
    main()