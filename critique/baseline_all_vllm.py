# ...existing code...
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from utils import resolve_io_keys

# Templates -------------------------------------------------------------------
# StrategyQA / yes-no
YES_NO_VANILLA_UNC = """Answer the following yes/no question and provide your uncertainty score. Your response should end with 'The answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is yes or no, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {}
"""

YES_NO_VANILLA_CONF = """Answer the following yes/no question and provide your confidence score. Your response should end with 'The answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is yes or no, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {}
"""

# Trivia / Hotpot open_end QA
OPEN_VANILLA_UNC = """Answer the following question and provide your uncertainty score. Your response should end with 'The answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is the direct answer to the question, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {}
"""
OPEN_VANILLA_CONF = """Answer the following question and provide your confidence score. Your response should end with 'The answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is the direct answer to the question, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {}
"""

# Multiple-choice (comparisonqa)
MC_VANILLA_UNC = """Answer the following multiple choice question. Select only one correct answer from the choices and give your uncertainty score about this question. Your response should end with 'The answer is [option_letter], and the uncertainty is [uncertainty_percentage]%' where the [option_letter] is one of A, B, C and D, and the uncertainty percentage should be a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {} A. {}. B. {}. C. {}. D. {}
"""
MC_VANILLA_CONF = """Answer the following multiple choice question. Select only one correct answer from the choices and give your confidence score about this question. Your response should end with 'The answer is [option_letter], and the confidence is [confidence_percentage]%' where the [option_letter] is one of A, B, C and D, and the confidence percentage should be a number between 0 and 100, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {} A. {}. B. {}. C. {}. D. {}
"""

# Math prompts (math500 / math_perturb)
MATH_VANILLA_UNC = """Answer the following math question and provide your uncertainty score. Your response should end with 'The answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is the final answer, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question.

Requirements:
1. Step-by-step reasoning: Provide a clear, step-by-step derivation of the answer, showing all necessary calculations and logical steps in the solution.
2. Final answer: Present the final answer [your_answer] in LaTeX format, wrapped in '\\boxed{{}}', ensuring the answer is concise and follows mathematical notation standards.
3. Uncertainty level: After the final answer, state your uncertainty in the solution as a percentage. If you are not sure, you should give a higher uncertainty percentage.
4. Concise language: Avoid redundancy and focus directly on the problem's core.
5. Format consistency: Ensure the answer format matches the problem's requirements (e.g., fractions, radicals, intervals).

Question: {}
Solution: """

MATH_VANILLA_CONF = """Answer the following math question and provide your confidence score. Your response should end with 'The answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is the final answer, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your answer.

Requirements:
1. Step-by-step reasoning: Provide a clear, step-by-step derivation of the answer, showing all necessary calculations and logical steps in the solution.
2. Final answer: Present the final answer [your_answer] in LaTeX format, wrapped in '\\boxed{{}}', ensuring the answer is concise and follows mathematical notation standards.
3. Confidence level: After the final answer, state your confidence in the solution as a percentage. If you are not sure, you should give a lower confidence percentage.
4. Concise language: Avoid redundancy and focus directly on the problem's core.
5. Format consistency: Ensure the answer format matches the problem's requirements (e.g., fractions, radicals, intervals).

Question: {}
Solution: """

# Task defaults ----------------------------------------------------------------
TASK_DEFAULT_INPUT = {
    "strategyqa": "../strategyqa_benchmark/dev.json",
    "comparisonqa": "../comparisonqa_benchmark/comparisonqa_test.json",
    "hotpotqa": "../hotpotqa_benchmark/hotpot_dev_fullwiki_v1.json",
    "triviaqa": "../triviaqa-benchmark/unfiltered-web-dev.json",
    "math500": "../math_benchmark/test500.json",
    "math_perturb": "../math_perturb_benchmark/math_perturb.json"
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
def prepare_prompts_comparisonqa(data, mode):
    """ComparisonQA: prepare high/low prompts and return prompt_list containing ('high_{idx}'/'low_{idx}')."""
    prompt_list = []
    for idx, item in enumerate(tqdm(data, desc="Preparing comparisonqa prompts")):
        high_q = item.get("high_question", {})
        low_q = item.get("low_question", {})

        # choose template
        template = MC_VANILLA_UNC if mode == "unc" else MC_VANILLA_CONF

        high_opts = high_q.get("options", {})
        low_opts = low_q.get("options", {})

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


def prepare_prompts_for_task(task, data, mode):
    """
    return (data, prompt_list) prompt_list format: list[(prompt_str, request_id)]
    request_id format "item_{idx}"
    """
    if task == "comparisonqa":
        return prepare_prompts_comparisonqa(data, mode)

    prompt_list = []
    for idx, item in enumerate(tqdm(data, desc=f"Preparing prompts for {task}")):
        # try resolve input key (if dataset fields vary)
        input_key, _ = resolve_io_keys(task)

        q = item.get(input_key, "")

        if task in ("math500", "math_perturb"):
            prompt = MATH_VANILLA_UNC.format(q) if mode == "unc" else MATH_VANILLA_CONF.format(q)

        elif task in ("triviaqa", "hotpotqa"):
            prompt = OPEN_VANILLA_UNC.format(q) if mode == "unc" else OPEN_VANILLA_CONF.format(q)

        else:  # strategyqa / default yes-no style
            prompt = YES_NO_VANILLA_UNC.format(q) if mode == "unc" else YES_NO_VANILLA_CONF.format(q)

        item["prompt"] = prompt
        prompt_list.append((prompt, f"item_{idx}"))
    return data, prompt_list

# add results back ------------------------------------------------------------
def add_results_generic(data, results, task):
    if task == "comparisonqa":
        return add_results_comparisonqa(data, results)
    for response, request_id in results:
        if request_id is None:
            continue
        try:
            idx = int(request_id.split("_", 1)[1])
            if 0 <= idx < len(data):
                data[idx]["response"] = response
            for key in ('decomposition', 'evidence', 'context', 'supporting_facts', 'SearchResults', 'EntityPages'):
                data[idx].pop(key, None)
        except Exception as e:
            print(f"Could not parse request_id '{request_id}': {e}")

def add_results_comparisonqa(data, results):
    """Attach high_response/low_response back to data using request_id 'high_{idx}' / 'low_{idx}'."""
    for response, request_id in results:
        prefix, idx = request_id.split("_", 1)
        idx = int(idx)
        if 0 <= idx < len(data):
            if prefix == "high":
                data[idx]["high_response"] = response
            elif prefix == "low":
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
    parser.add_argument("--input_file", type=str, default=None, help="Path to input JSON file, which is only needed when conducting self_critique.")
    parser.add_argument("--num_processes", type=int, default=300)
    parser.add_argument("--mode", type=str, default="unc", choices=["unc", "conf"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.port}/v1"

    input_file = args.input_file or TASK_DEFAULT_INPUT.get(args.task)

    data = load_json(input_file)
    data = data[:10]

    data, prompt_list = prepare_prompts_for_task(args.task, data, args.mode)

    client = ParallelChatClient(
        api_key=openai_api_key,
        api_base=openai_api_base,
        num_processes=args.num_processes
    )

    model_name = client.client.models.list().data[0].id
    print(f"Using model: {model_name}")

    results = client.generate_batch(prompt_list)

    add_results_generic(data, results, task=args.task)

    # default output path if not provided
    name = model_name.split("/")[-1].replace(".", "").replace("-", "_").replace(":", "_")
    default_out = f"../experiments/{args.task}/{args.task}_test_{name}_parallel_vanilla_{args.mode}.json"
    out_path = args.output_path or default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_json(data, out_path)
    print(f"Results saved to: {out_path}")