import re
import numpy as np
import string

def resolve_io_keys(task_name: str):
    """
    Return question_key based on task_name.
    Raise ValueError if detection fails, indicating that the user must configure it in the mapping.
    """
    mapping = {
        "strategyqa": "question",
        "triviaqa": "Question",
        "hotpotqa": "question",
        "math500": "problem",
        "math_perturb": "problem"
    }

    t = task_name.lower()
    if t in mapping:
        return mapping[t]

    raise ValueError(f"can not identify the question of task '{task_name}'")



def extract_prediction_mc(output, options):
    answer = []
    
    pattern = r'([A-D])(?=\s|$|[.,:()\[\]\'"])'
    searches = [
        r"(?<=answer is )([^.,]*)(?:[.,]*|$)",
        r"(?<=answer)([^.,]*)(?:[.,]*|$)"
    ]

    for regex in searches:
        matches = re.findall(regex, output, re.IGNORECASE)
        for match_text in matches:
            match_text = match_text.strip().strip(':').strip('[]()')
            match = re.search(pattern, match_text)
            if match:
                prediction = match.group(1)
                if prediction in ["A", "B", "C", "D"]:
                    answer.append(prediction)

    if len(answer) > 0:
        return answer[-1]

    matches = output.split("\n")[-1]
    match = re.search(pattern, matches)
    if match:
        prediction = match.group(1)
        if prediction in ["A", "B", "C", "D"]:
            return prediction
            
    # print(output)
    return None


def extract_prediction_open(output):
    searches = [
        r'answer is:?\s*(.*?)(?:,?\s*and the)',
        r'answer is:?\s*(.*?)(?:,?\s*\.)',
        r'answer: \s*(.*?)(?:,?\s*\.)',
        r'answer would be\s*"(.*?)"',
        r'answer is:?\s*(.*?)(?:,?\s*,)',
        r'answer:?\*\*:?\s*(.*?)(?:,?\s*(\*\*)|\.)',
        # r'answer:?\s*(.*?)(?:,?\s*\.)'
    ]

    for regex in searches:  
        match = re.search(regex, output, re.IGNORECASE)
        if match:
            pred = match.group(1).strip()
            t = re.search(r'[\'"](.*?)[\"\']', pred)
            if t:
                pred = t.group(1).strip()
            t = pred.split(',')
            if len(t) > 1:
                if len(t[1].split())>3:
                    pred = t[0]
            exclude = set(string.punctuation)
            pred = ''.join(ch for ch in pred if ch not in exclude).strip()
            if pred.lower().startswith("yes"):
                return "yes"
            elif pred.lower().startswith("no"):
                return "no"
            return pred

    return None


# def extract_prediction_gsm8k(output):
#     searches = [
#         r"The answer is (.*?), and the ",
#         r"The answer is (.*?)\."
#     ]

#     for regex in searches:  
#         match = re.search(regex, output, re.IGNORECASE)
#         if match:
#             answer = match.group(1).strip()
#             answer = re.sub(r'^\s*["\']|["\']\s*$', '', answer).strip()
#             answer = re.sub(r'[,\s]', '', answer)
#             answer = re.findall(r'\d+', answer)
#             if answer:
#                 answer = answer[0]
#                 try:
#                     return int(answer)
#                 except:
#                     continue
#     return None


def extract_box(output):
    searches = [
        r'\\boxed\{(.*?)\}\$',
        r'\\boxed\{(.*?)\}\.',
        r'\\boxed\{(.*?)\}\,',
        r'\\boxed\{(.*?)\}\n',
        r'\\boxed\{(.*?)\}\\\)',
        r'\\boxed\{(.*?)\} \\\)',
        r'\\boxed\{(.*?)\} \\\]',
        r'\\boxed\{(.*?)\} ',
        r'\\boxed\{(.*?)\}\\',
        r'\\boxed\{(.*?)\}'
    ]

    for regex in searches:  
        match = re.search(regex, output, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            return answer
        continue

    searches = [
        r"The answer is (.*?), and the ",
        r"The answer is (.*?)\."
    ]

    for regex in searches:  
        match = re.search(regex, output, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if answer.startswith('[') and answer.endswith(']'):
                return answer[1:-1].strip()
            return answer
        continue
            

    return None


def extract_uncertainty(output):
    match = re.search(r'uncertainty is\s*\[?(\d+(\.\d+)?)\]?%', output)
    if match:
        return float(match.group(1)) / 100.0
    
    matches = re.findall(r'(\d+(\.\d+)?)\]?%', output)
    if matches:
        return float(matches[-1][0]) / 100.0

    # print(output)
    return None


def extract_confidence(output):
    match = re.search(r'confidence is\s*\[?(\d+(\.\d+)?)\]?%', output)
    if match:
        return float(match.group(1)) / 100.0

    match = re.search(r'confidence in this answer is\s*\[?(\d+(\.\d+)?)\]?%', output)
    if match:
        return float(match.group(1)) / 100.0
    
    matches = re.findall(r'(\d+(\.\d+)?)\]?%', output)
    if matches:
        return float(matches[-1][0]) / 100.0

    # print(output)
    return None

