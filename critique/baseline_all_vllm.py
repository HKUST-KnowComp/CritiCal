import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from utils import resolve_io_keys
from prompt import *


# Task defaults ----------------------------------------------------------------
TASK_DEFAULT_INPUT = {
    "strategyqa": "../benchmark/strategyqa/dev.json",
    "comparisonqa": "../benchmark/comparisonqa/comparisonqa_test.json",
    "hotpotqa": "../benchmark/hotpotqa/hotpot_dev_fullwiki_v1.json",
    "triviaqa": "../benchmark/triviaqa/unfiltered-web-dev.json",
    "math500": "../benchmark/math500/test500.json",
    "math_perturb": "../benchmark/math_perturb/math_perturb.json"
}


# IO helpers ------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "Data" in data:
            return data["Data"]
        return data

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


# prepare prompts per task ----------------------------------------------------
def prepare_prompts_for_task(task, data, mode, run_mode="vanilla"):
    """
    return (data, prompt_list) prompt_list format: list[(prompt_str, request_id)]
    request_id format "item_{idx}"
    """
    if task == "comparisonqa":
        return prepare_prompts_comparisonqa(data, mode, run_mode)

    prompt_list = []
    for idx, item in enumerate(tqdm(data, desc=f"Preparing prompts for {task}")):
        # try resolve input key (if dataset fields vary)
        input_key, _ = resolve_io_keys(task)

        q = item.get(input_key, "")

        if run_mode == "self_critique":
            # use self-critique templates and include initial response from input_file
            initial = item.get("response", "")
            if task in ("math500", "math_perturb"):
                template = MATH_SELF_CRITIQUE_UNC if mode == "unc" else MATH_SELF_CRITIQUE_CONF
                prompt = template.format(q, initial)
            elif task in ("triviaqa", "hotpotqa"):
                template = OPEN_SELF_CRITIQUE_UNC if mode == "unc" else OPEN_SELF_CRITIQUE_CONF
                prompt = template.format(q, initial)
            else:  # strategyqa / default yes-no style
                template = YES_NO_SELF_CRITIQUE_UNC if mode == "unc" else YES_NO_SELF_CRITIQUE_CONF
                prompt = template.format(q, initial)

            item["self_critique_prompt"] = prompt
        else:
            # vanilla behavior
            if task in ("math500", "math_perturb"):
                prompt = MATH_VANILLA_UNC.format(q) if mode == "unc" else MATH_VANILLA_CONF.format(q)
            elif task in ("triviaqa", "hotpotqa"):
                prompt = OPEN_VANILLA_UNC.format(q) if mode == "unc" else OPEN_VANILLA_CONF.format(q)
            else:  # strategyqa / default yes-no style
                prompt = YES_NO_VANILLA_UNC.format(q) if mode == "unc" else YES_NO_VANILLA_CONF.format(q)

            item["prompt"] = prompt

        prompt_list.append((prompt, f"item_{idx}"))
    return data, prompt_list


def prepare_prompts_comparisonqa(data, mode, run_mode="vanilla"):
    """ComparisonQA: prepare high/low prompts and return prompt_list containing ('high_{idx}'/'low_{idx}')."""
    prompt_list = []
    for idx, item in enumerate(tqdm(data, desc="Preparing comparisonqa prompts")):
        high_q = item.get("high_question", {})
        low_q = item.get("low_question", {})

        # choose template based on run_mode & mode
        if run_mode == "self_critique":
            template = MC_SELF_CRITIQUE_UNC if mode == "unc" else MC_SELF_CRITIQUE_CONF
        else:
            template = MC_VANILLA_UNC if mode == "unc" else MC_VANILLA_CONF

        high_opts = high_q.get("options", {})
        low_opts = low_q.get("options", {})

        if run_mode == "self_critique":
            # expect previous-round responses to be present as high_response / low_response
            high_initial = item.get("high_response", "")
            low_initial = item.get("low_response", "")

            high_prompt = template.format(
                high_q.get("question", ""),
                high_opts.get("A", ""), high_opts.get("B", ""), high_opts.get("C", ""), high_opts.get("D", ""),
                high_initial
            )
            low_prompt = template.format(
                low_q.get("question", ""),
                low_opts.get("A", ""), low_opts.get("B", ""), low_opts.get("C", ""), low_opts.get("D", ""),
                low_initial
            )
            item["high_self_critique_prompt"] = high_prompt
            item["low_self_critique_prompt"] = low_prompt

        else:
            high_prompt = template.format(
                high_q.get("question", ""),
                high_opts.get("A", ""), high_opts.get("B", ""), high_opts.get("C", ""), high_opts.get("D", "")
            )
            low_prompt = template.format(
                low_q.get("question", ""),
                low_opts.get("A", ""), low_opts.get("B", ""), low_opts.get("C", ""), low_opts.get("D", "")
            )
            item["high_prompt"] = high_prompt
            item["low_prompt"] = low_prompt


        prompt_list.append((high_prompt, f"high_{idx}"))
        prompt_list.append((low_prompt, f"low_{idx}"))
    return data, prompt_list


# add results back ------------------------------------------------------------
def add_results_generic(data, results, task, run_mode="vanilla"):
    if task == "comparisonqa":
        return add_results_comparisonqa(data, results, run_mode)
    for response, request_id in results:
        if request_id is None:
            continue
        try:
            idx = int(request_id.split("_", 1)[1])
            if 0 <= idx < len(data):
                if run_mode == "self_critique":
                    data[idx]["self_critique_response"] = response
                else:
                    data[idx]["response"] = response
            for key in ('decomposition', 'evidence', 'context', 'supporting_facts', 'SearchResults', 'EntityPages'):
                data[idx].pop(key, None)
        except Exception as e:
            print(f"Could not parse request_id '{request_id}': {e}")


def add_results_comparisonqa(data, results, run_mode="vanilla"):
    """Attach high_response/low_response back to data using request_id 'high_{idx}' / 'low_{idx}'."""
    for response, request_id in results:
        prefix, idx = request_id.split("_", 1)
        idx = int(idx)
        if 0 <= idx < len(data):
            if prefix == "high":
                if run_mode == "self_critique":
                    data[idx]["high_self_critique_response"] = response
                else:
                    data[idx]["high_response"] = response
            elif prefix == "low":
                if run_mode == "self_critique":
                    data[idx]["low_self_critique_response"] = response
                else:
                    data[idx]["low_response"] = response


# Parallel client (reuse existing logic) -------------------------------------
class ParallelChatClient:
    def __init__(self, api_key, api_base, num_processes=4):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=600
        )
        self.num_processes = num_processes
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.ERROR)

    def _generate_single(self, prompt, request_id):
        try:
            model_name = self.client.models.list().data[0].id
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                stream=False,
                max_tokens=4096,
            )
            response = chat_completion.choices[0].message.content
            return (response, request_id)
        except Exception as e:
            self.logger.error(f"Error generating for request {request_id}: {str(e)}")
            return (str(e), request_id)

    def generate_batch(self, prompt_list):
        results = []
        future_to_idx = {}
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            for idx, (prompt, request_id) in enumerate(prompt_list):
                future = executor.submit(self._generate_single, prompt, request_id)
                future_to_idx[future] = idx

            with tqdm(total=len(future_to_idx), desc="Generating responses") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        request_id = prompt_list[idx][1]
                        self.logger.error(f"Error in batch generation for request {request_id}: {str(e)}")
                        results.append((str(e), request_id))
                    pbar.update(1)

        # try restore original order
        if all(r[1] is not None for r in results):
            results.sort(key=lambda x: int(x[1].split("_", 1)[1]))
        return results

# CLI ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="strategyqa",
                        choices=["strategyqa", "comparisonqa", "hotpotqa", "triviaqa", "math500", "math_perturb"])
    parser.add_argument("--input_file", type=str, default=None, help="Path to input JSON file, which is only needed when conducting self_critique (contains first-round responses).")
    parser.add_argument("--num_processes", type=int, default=300)
    parser.add_argument("--mode", type=str, default="unc", choices=["unc", "conf"])
    parser.add_argument("--run_mode", type=str, default="vanilla", choices=["vanilla", "self_critique"], help="Choose 'vanilla' or 'self_critique' flow.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    # For self-critique we require an input_file that contains the first-round responses.
    if args.run_mode == "self_critique" and not args.input_file:
        raise SystemExit("Error: --input_file must be provided in self_critique mode (file should contain first-round responses).")

    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.port}/v1"

    # choose input source: if vanilla, allow default dataset; if self_critique, use provided input_file
    input_file = args.input_file if args.run_mode == "self_critique" else TASK_DEFAULT_INPUT.get(args.task)

    data = load_json(input_file)
    data = data[:10]

    data, prompt_list = prepare_prompts_for_task(args.task, data, args.mode, run_mode=args.run_mode)

    client = ParallelChatClient(
        api_key=openai_api_key,
        api_base=openai_api_base,
        num_processes=args.num_processes
    )

    model_name = client.client.models.list().data[0].id
    print(f"Using model: {model_name}")

    results = client.generate_batch(prompt_list)

    add_results_generic(data, results, task=args.task, run_mode=args.run_mode)

    # default output path if not provided
    name = model_name.split("/")[-1].replace(".", "").replace("-", "_").replace(":", "_")
    default_out = f"../experiments/{args.task}/{args.task}_test_{name}_parallel_{args.run_mode}_{args.mode}.json"
    out_path = args.output_path or default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_json(data, out_path)
    print(f"Results saved to: {out_path}")