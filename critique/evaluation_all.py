import json
import re
import string
import argparse
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from netcal.metrics import ECE
import os

from utils import (
    extract_prediction_open,
    extract_prediction_mc,
    extract_box,
    extract_uncertainty,
    extract_confidence,
    is_equiv_500,
    is_equiv_perturb
)

def normalize_answer(s):
    if s is None:
        return None
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(ground_truth) in normalize_answer(prediction))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores) if scores else 0.0

def compute_ece_auroc(confidences, corrects):
    confidences_np = np.array(confidences)
    corrects_np = np.array(corrects)
    if len(confidences_np) == 0:
        return float('nan'), float('nan')
    if len(np.unique(corrects_np)) < 2:
        auroc = float('nan')
    else:
        try:
            auroc = roc_auc_score(corrects_np, confidences_np)
        except Exception:
            auroc = float('nan')
    ece_calculator = ECE(10)
    try:
        ece = ece_calculator.measure(confidences_np, corrects_np)
    except Exception:
        ece = float('nan')
    return ece, auroc

def evaluate_triviaqa(data, input_type):
    fail_list = []
    accuracy = 0.0
    confidences = []
    corrects = []
    fail_count_pred = 0
    fail_count_conf = 0
    total = 0

    for item in tqdm(data, desc="Evaluating triviaqa"):
        response = item.get("response", "")
        gold_answers = item.get("Answer", {}).get("NormalizedAliases", [])

        pred_answer = extract_prediction_open(response)
        if input_type == 'uncertainty':
            unc = extract_uncertainty(response)
            conf = 1.0 - unc if unc is not None else None
        else:
            conf = extract_confidence(response)

        if pred_answer is None:
            fail_count_pred += 1
            response_t = response.split(", and the uncertainty")[0].split("The uncertainty")[0]
            if len(response_t.split(" ")) > 30:
                pred_answer = ""
                fail_list.append((response, conf))
            else:
                pred_answer = response_t

        em = metric_max_over_ground_truths(exact_match_score, pred_answer, gold_answers)

        if conf is None:
            fail_count_conf += 1
            conf = 1.0
            if em == 1:
                conf = 0.0
            fail_list.append((response, conf))

        accuracy += float(em)
        corrects.append(em)
        confidences.append(conf)
        total += 1

    if total == 0:
        return {}
    accuracy /= total
    ece, auroc = compute_ece_auroc(confidences, corrects)
    fail_path = "../experiments/triviaqa/fail.json"
    os.makedirs(os.path.dirname(fail_path), exist_ok=True)
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)
    return {'em': accuracy*100, 'ece': ece, 'auroc': auroc, 'fail_count_pred': fail_count_pred, 'fail_count_conf': fail_count_conf, 'total': total}

def evaluate_hotpotqa(data, input_type):
    fail_list = []
    accuracy = 0.0
    total = 0
    confidences = []
    corrects = []
    fail_count_pred = 0
    fail_count_conf = 0

    for item in tqdm(data, desc="Evaluating hotpotqa"):
        response = item.get("response", "")
        gold_answer = item.get("answer", "")

        pred_answer = extract_prediction_open(response)
        if input_type == 'uncertainty':
            unc = extract_uncertainty(response)
            conf = 1.0 - unc if unc is not None else None
        else:
            conf = extract_confidence(response)

        if pred_answer is None:
            fail_count_pred += 1
            pred_answer = ""
            fail_list.append((response, conf))

        em = exact_match_score(pred_answer, gold_answer)

        if conf is None:
            fail_count_conf += 1
            conf = 1.0
            if em == 1:
                conf = 0.0
            fail_list.append((response, conf))

        # update
        accuracy += float(em)
        total += 1
        corrects.append(em)
        confidences.append(conf)

    ece, auroc = compute_ece_auroc(confidences, corrects)
    accuracy /= total
    summary = {'em': accuracy*100, 'ece': ece, 'auroc': auroc, 'fail_count_pred': fail_count_pred, 'fail_count_conf': fail_count_conf, 'total': total}

    fail_path = "../experiments/hotpotqa/fail.json"
    os.makedirs(os.path.dirname(fail_path), exist_ok=True)
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)
    return summary

def evaluate_strategyqa(data, input_type):
    fail_list = []
    accuracy = 0.0
    confidences = []
    corrects = []
    fail_count_pred = 0
    fail_count_conf = 0
    total = 0

    for item in tqdm(data, desc="Evaluating strategyqa"):
        response = item.get("response", "")
        gold_answer = item.get("answer")

        pred_raw = extract_prediction_open(response)
        pred_norm = normalize_answer(pred_raw)
        if pred_norm == "yes":
            pred = True
        elif pred_norm == "no":
            pred = False
        else:
            pred = None

        if input_type == 'uncertainty':
            unc = extract_uncertainty(response)
            conf = 1.0 - unc if unc is not None else None
        else:
            conf = extract_confidence(response)

        if pred is None:
            fail_count_pred += 1
            pred = ""
            fail_list.append((response, conf))

        acc = 1 if pred == gold_answer else 0

        if conf is None:
            fail_count_conf += 1
            conf = 1.0
            if acc == 1:
                conf = 0.0
            fail_list.append((response, conf))

        accuracy += float(acc)
        corrects.append(acc)
        confidences.append(conf)
        total += 1

    if total == 0:
        return {}
    accuracy /= total
    ece, auroc = compute_ece_auroc(confidences, corrects)
    fail_path = "../experiments/strategyqa/fail.json"
    os.makedirs(os.path.dirname(fail_path), exist_ok=True)
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)
    return {'acc': accuracy*100, 'ece': ece, 'auroc': auroc, 'fail_count_pred': fail_count_pred, 'fail_count_conf': fail_count_conf, 'total': total}

def evaluate_comparisonqa(data, input_type):
    fail_list = []
    label_dic = {"A": 0, "B": 1, "C": 2, "D": 3, None: 4}
    high_confidences = []
    low_confidences = []
    high_predict_label = []
    low_predict_label = []
    high_gold_label = []
    low_gold_label = []

    fail_count_high_acc = 0
    fail_count_low_acc = 0
    fail_count_high_conf = 0
    fail_count_low_conf = 0

    for line in tqdm(data, desc="Evaluating comparisonqa"):
        # high
        if line.get("high_response") and line["high_response"] != "":
            high_predict = extract_prediction_mc(line["high_response"], line["high_question"]["options"])
            high_gold = list(line["high_question"]["answer"].keys())[0]
            if high_predict is None:
                fail_list.append(line["high_response"])
                fail_count_high_acc += 1
            if input_type == 'uncertainty':
                high_unc = extract_uncertainty(line["high_response"])
                high_conf = 1.0 - high_unc if high_unc is not None else None
            else:
                high_conf = extract_confidence(line["high_response"])
            if high_conf is None:
                high_conf = 1.0
                if high_predict == high_gold:
                    high_conf = 0.0
                fail_list.append((line["high_response"], high_conf))
                fail_count_high_conf += 1
            high_confidences.append(high_conf)
            high_predict_label.append(label_dic.get(high_predict))
            high_gold_label.append(label_dic.get(high_gold))

        # low
        if line.get("low_response") and line["low_response"] != "":
            low_predict = extract_prediction_mc(line["low_response"], line["low_question"]["options"])
            low_gold = list(line["low_question"]["answer"].keys())[0]
            if low_predict is None:
                fail_list.append(line["low_response"])
                fail_count_low_acc += 1
            if input_type == 'uncertainty':
                low_unc = extract_uncertainty(line["low_response"])
                low_conf = 1.0 - low_unc if low_unc is not None else None
            else:
                low_conf = extract_confidence(line["low_response"])
            if low_conf is None:
                low_conf = 1.0
                if low_predict == low_gold:
                    low_conf = 0.0
                fail_list.append((line["low_response"], low_conf))
                fail_count_low_conf += 1
            low_confidences.append(low_conf)
            low_predict_label.append(label_dic.get(low_predict))
            low_gold_label.append(label_dic.get(low_gold))

    # For presentation, compute simple accuracies and calibration using labels
    high_acc = (sum(np.array(high_predict_label) == np.array(high_gold_label)) / len(high_gold_label)) if high_gold_label else float('nan')
    low_acc = (sum(np.array(low_predict_label) == np.array(low_gold_label)) / len(low_gold_label)) if low_gold_label else float('nan')

    high_corrects = (np.array(high_predict_label) == np.array(high_gold_label)) if high_gold_label else np.array([])
    low_corrects = (np.array(low_predict_label) == np.array(low_gold_label)) if low_gold_label else np.array([])

    high_ece, high_auroc = (float('nan'), float('nan'))
    low_ece, low_auroc = (float('nan'), float('nan'))
    try:
        high_ece, high_auroc = compute_ece_auroc(high_confidences, high_corrects) if len(high_confidences) else (float('nan'), float('nan'))
        low_ece, low_auroc = compute_ece_auroc(low_confidences, low_corrects) if len(low_confidences) else (float('nan'), float('nan'))
    except Exception:
        pass

    fail_path = "../experiments/comparisonqa/fail.json"
    os.makedirs(os.path.dirname(fail_path), exist_ok=True)
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)

    return {
        'high_acc': high_acc*100, 'low_acc': low_acc*100,
        'high_ece': high_ece, 'low_ece': low_ece,
        'high_auroc': high_auroc, 'low_auroc': low_auroc,
        'fail_count_high_acc': fail_count_high_acc,
        'fail_count_low_acc': fail_count_low_acc,
        'fail_count_high_conf': fail_count_high_conf,
        'fail_count_low_conf': fail_count_low_conf,
        'total_high': len(high_gold_label),
        'total_low': len(low_gold_label),
        'avg_acc': ((high_acc + low_acc)/2)*100,
        'avg_ece': (high_ece + low_ece)/2,
        'avg_auroc': (high_auroc + low_auroc)/2
    }

def evaluate_math500(data, input_type):
    fail_list = []
    accuracy = 0.0
    confidences = []
    corrects = []
    fail_count_pred = 0
    fail_count_conf = 0
    total = 0

    for item in tqdm(data, desc="Evaluating math500"):
        response = item.get("response", "")
        gold_answer = extract_box(item.get("solution")) or ""
        if gold_answer == "":
            fail_list.append((item.get("solution"), "gold"))

        pred_answer = extract_box(response)
        if pred_answer is None:
            fail_count_pred += 1
            pred_answer = ""
            fail_list.append((response, "pred"))

        equiv = is_equiv_500(pred_answer, gold_answer)
        acc = 1 if equiv else 0

        response = response.replace("\\", "").replace("boxed", "").replace("}", "")
        if input_type == 'uncertainty':
            unc = extract_uncertainty(response)
            conf = 1.0 - unc if unc is not None else None
        else:
            conf = extract_confidence(response)

        if conf is None:
            fail_count_conf += 1
            conf = 1.0
            if acc == 1:
                conf = 0.0
            fail_list.append((response, conf))

        accuracy += float(acc)
        corrects.append(acc)
        confidences.append(conf)
        total += 1

    if total == 0:
        return {}
    accuracy /= total
    ece, auroc = compute_ece_auroc(confidences, corrects)
    fail_path="../experiments/math500/fail.json"
    os.makedirs(os.path.dirname(fail_path), exist_ok=True)
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)
    return {'acc': accuracy*100, 'ece': ece, 'auroc': auroc, 'fail_count_pred': fail_count_pred, 'fail_count_conf': fail_count_conf, 'total': total}

def evaluate_math_perturb(data, input_type):
    fail_list = []
    accuracy = 0.0
    confidences = []
    corrects = []
    fail_count_pred = 0
    fail_count_conf = 0
    total = 0

    predict_all = []
    for item in data:
        if item["original_split"] == "test":
            predict_all.append(item)

    for item in tqdm(predict_all, desc="Evaluating math_perturb"):
        response = item.get("response", "")
        gold_answer = str(item.get("answer"))

        pred_answer = extract_box(response)
        if pred_answer is None:
            fail_count_pred += 1
            pred_answer = ""
            fail_list.append((response, "pred"))

        equiv = is_equiv_perturb(pred_answer, gold_answer)
        acc = 1 if equiv else 0

        response = response.replace("\\", "").replace("boxed", "").replace("}", "")
        if input_type == 'uncertainty':
            unc = extract_uncertainty(response)
            conf = 1.0 - unc if unc is not None else None
        else:
            conf = extract_confidence(response)

        if conf is None:
            fail_count_conf += 1
            conf = 1.0
            if acc == 1:
                conf = 0.0
            fail_list.append((response, conf))

        accuracy += float(acc)
        corrects.append(acc)
        confidences.append(conf)
        total += 1

    if total == 0:
        return {}
    accuracy /= total
    ece, auroc = compute_ece_auroc(confidences, corrects)
    fail_path = "../experiments/math_perturb/fail.json"
    os.makedirs(os.path.dirname(fail_path), exist_ok=True)
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)
    return {'acc': accuracy*100, 'ece': ece, 'auroc': auroc, 'fail_count_pred': fail_count_pred, 'fail_count_conf': fail_count_conf, 'total': total}


# CLI ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="strategyqa",
                        choices=["strategyqa", "comparisonqa", "hotpotqa", "triviaqa", "math500", "math_perturb"])
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--input_type", type=str, default="uncertainty", choices=["uncertainty", "confidence"])
    args = parser.parse_args()

    file_path = args.file_path
    if not file_path:
        raise SystemExit("Please provide --file_path")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise SystemExit(f"Can not read JSON file: {e}")

    if isinstance(data, dict) and "Data" in data:
        data = data["Data"]

    task_to_func = {
        "triviaqa": evaluate_triviaqa,
        "hotpotqa": evaluate_hotpotqa,
        "strategyqa": evaluate_strategyqa,
        "comparisonqa": evaluate_comparisonqa,
        "math500": evaluate_math500,
        "math_perturb": evaluate_math_perturb,
    }

    evaluate_func = task_to_func.get(args.task)
    res = evaluate_func(data, args.input_type) if evaluate_func else None

    print("\n--- Summary ---")
    print(res)