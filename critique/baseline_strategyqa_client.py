from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import logging
import argparse
import json
import os


pure_unc_prompt = """Answer the following yes/no question and provide your uncertainty score. Your response should end with 'The answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is yes or no, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {}
"""

pure_conf_prompt = """Answer the following yes/no question and provide your confidence score. Your response should end with 'The answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is yes or no, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {}
"""

clarify_unc_prompt = """Uncertainty is a property of the model's predictive distribution, capturing the degree of variability or unpredictability given a particular input. In contrast, confidence reflects the model's belief in the correctness of a particular prediction.

Answer the following yes/no question and provide your uncertainty score. Your response should end with 'The answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is yes or no, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {}
"""

clarify_conf_prompt = """Uncertainty is a property of the model's predictive distribution, capturing the degree of variability or unpredictability given a particular input. In contrast, confidence reflects the model's belief in the correctness of a particular prediction.

Answer the following yes/no question and provide your confidence score. Your response should end with 'The answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is yes or no, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {}
"""

self_critique_unc_prompt = """You previously answered the following yes/no question with the response ended with: 'The answer is [initial_answer], and the uncertainty is [initial_uncertainty]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your uncertainty score. Answer the question and provide a new uncertainty score. Your response should end with 'The refined answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is yes or no, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure about the question, you should give a higher uncertainty percentage. 

Question: {}
Initial response: {}
"""

self_critique_conf_prompt = """You previously answered the following yes/no question with the response ended with: 'The answer is [initial_answer], and the confidence is [initial_confidence]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your confidence score. Answer the question and provide a new confidence score. Your response should end with 'The refined answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is yes or no, and the confidence percentage is a number between 0 and 100, indicating how uncertain you are about your refined answer. If you are not sure about your refined answer, you should give a lower confidence percentage. 

Question: {}
Initial response: {}
"""

def load_data(file_path):
    """Load JSON data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_prompts(data, mode):
    """Prepare prompts for HotpotQA data (open-ended QA)"""
    prompt_list = []
    for idx, item in enumerate(tqdm(data, desc="Preparing prompts")):
        question = item.get("question")
        
        if mode == "unc":
            prompt = pure_unc_prompt
        else:
            prompt = pure_conf_prompt
        item['prompt'] = prompt.format(question)


        # if mode == "unc":
        #     prompt = self_critique_unc_prompt
        # else:
        #     prompt = self_critique_conf_prompt
        # response = item.get("response")
        # # item['prompt'] = prompt.format(question, response)
        # item['self_critique_prompt'] = prompt.format(question, response)

        prompt_list.append((item['prompt'], f"item_{idx}"))
        # prompt_list.append((item['self_critique_prompt'], f"item_{idx}"))

    return data, prompt_list

class ParallelChatClient:
    def __init__(self, api_key, api_base, num_processes=4):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=300
        )
        self.num_processes = num_processes
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.ERROR)

    def _generate_single(self, prompt, request_id):
        """Generate response for a single prompt"""
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
        """Perform parallel inference on a list of prompts"""
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
        
        # Sort results back to original order
        if all(r[1] is not None for r in results):
            results.sort(key=lambda x: int(x[1].split('_')[1]))

        return results

def add_results_to_data(data, results):
    """Add results back to the original data structure"""
    for response, request_id in results:
        if request_id is None: continue
        try:
            idx = int(request_id.split('_')[1])
            if idx < len(data):
                data[idx]['response'] = response
                # data[idx]['self_critique_response'] = response
                if 'decomposition' in data[idx]:
                    del data[idx]['decomposition']
                if 'evidence' in data[idx]:
                    del data[idx]['evidence']
        except (ValueError, IndexError) as e:
            print(f"Could not parse request_id '{request_id}': {e}")


def save_results(results, output_path):
    """保存结果到文件"""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 8010 meta-llama/Llama-3.2-3B-Instruct
    # meta-llama/Llama-3.1-70B-Instruct
    # 8021 meta-llama/Llama-3.1-8B-Instruct
    # 8025 Qwen/Qwen3-8B
    # 8023 deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # 8022 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # 8020 Qwen/Qwen2.5-7B-Instruct
    # 8024 mistralai/Mistral-7B-Instruct-v0.3

    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file', type=str, default="../strategyqa_benchmark/train.json")
    parser.add_argument('--input_file', type=str, default="../strategyqa_benchmark/dev.json")
    # parser.add_argument('--input_file', type=str, default="../experiments/critique_cpu8/strategyqa/strategyqa_dev_Llama_31_8B_Instruct_parallel_pure_conf.json")
    parser.add_argument('--num_processes', type=int, default=300)
    parser.add_argument('--mode', type=str, default="unc", choices=["unc", "conf"])
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output_path', type=str, help="Path to save the output predictions.")
    args = parser.parse_args()

    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.port}/v1"
    
    data = load_data(args.input_file)
    # data = data[:10]
    
    data, prompt_list = prepare_prompts(data, args.mode)

    chat_client = ParallelChatClient(
        api_key=openai_api_key,
        api_base=openai_api_base,
        num_processes=args.num_processes
    )

    model_name = chat_client.client.models.list().data[0].id
    print(f"Using model: {model_name}")

    results = chat_client.generate_batch(prompt_list)

    add_results_to_data(data, results)

    name = model_name.split('/')[-1].replace('.', '').replace('-', '_').replace(':', '_')
    # output_path = f'../experiments/critique_cpu8/train/strategyqa_train_{name}_parallel_pure_{args.mode}.json'
    # output_path = f'../experiments/critique_cpu8/train_dpo/dpo_strategyqa_{name}_pure_{args.mode}.json'
    output_path = f'../experiments/critique_cpu8/sft/strategyqa/strategyqa_dev_{name}_parallel_pure_{args.mode}_28_30.json'
    # output_path = f'../experiments/critique_cpu8/strategyqa/strategyqa_dev_{name}_parallel_pure_{args.mode}.json'
    # output_path = f'../experiments/critique_cpu8/strategyqa/strategyqa_dev_{name}_parallel_pure_conf_self_critique1.json'

    if args.output_path:
        output_path = args.output_path

    save_results(data, output_path)
    print(f"Results saved to: {output_path}")
