import re
import numpy as np
import string
import regex

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



def _fix_fracs_500(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b_500(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units_500(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt_500(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string_500(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units_500(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt_500(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs_500(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b_500(string)

    return string

def is_equiv_500(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string_500(str1)
        ss2 = _strip_string_500(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2



def _fix_fracs_perturb(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b_perturb(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt_perturb(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan_perturb(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string

def _fix_unicode_perturb(string):
    # for debugging.
    before = string

    # square root 
    pattern = re.compile(r'√(\([^()]*\)|[A-Za-z0-9]+)')
    string = pattern.sub(lambda m: r'\sqrt{' + m.group(1) + '}', string)

    # cube root
    pattern = re.compile(r'∛(\([^()]*\)|[A-Za-z0-9]+)')
    string = pattern.sub(lambda m: r'\sqrt[3]{' + m.group(1) + '}', string)

    # other fonts of digits
    math_sans_bold_digits = {
        '𝟬': '0', '𝟭': '1', '𝟮': '2', '𝟯': '3', '𝟰': '4',
        '𝟱': '5', '𝟲': '6', '𝟳': '7', '𝟴': '8', '𝟵': '9',

        '𝟢': '0', '𝟣': '1', '𝟤': '2', '𝟥': '3', '𝟦': '4',
        '𝟧': '5', '𝟨': '6', '𝟩': '7', '𝟪': '8', '𝟫': '9',
    }
    for unicode_digit, ascii_digit in math_sans_bold_digits.items():
        string = string.replace(unicode_digit, ascii_digit)

    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3',
        '₄': '4', '₅': '5', '₆': '6', '₇': '7',
        '₈': '8', '₉': '9', 'ₙ': 'n'
    }
    for subchar, digit in subscript_map.items():
        string = string.replace(subchar, f"_{{{digit}}}")

    # other replacements
    replacements = {
        '²': '^{2}',
        '³': '^{3}',
        'ⁿ': '^{n}',
        'π': '\\pi ',
        '∞': '\\infty ',
        '⎣': '\\lfloor ',
        '⎦': '\\rfloor ',
        '–': '-', ## (en dash) U+2013 to (hyphen) U+002D
        '−': '-', ## (minus) U+2212 to (hyphen) U+002D
        '∪': '\\cup ',
        '∩': '\\cap ',
        '·': '\\cdot ',
        '×': '\\times ',
        ' ': ' ',
        '⁄': '/',
        '\xa0': ' ',
        '½': '\\frac{1}{2}',
        '∏': '\\prod ',
        '∑': '\\sum ',
    }
    
    for unicode_char, latex_equiv in replacements.items():
        string = string.replace(unicode_char, latex_equiv)

    if before != string:
        print(f"DEBUG: Unicode conversion: {before} -> {string}", flush=True)
    return string


def strip_string_perturb(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = re.sub(r'(?<!\\)\\ ', '', string) # remove "\\ " but not "\\\\ ".
    string = string.replace("\\,", "")
    string = string.replace("\\:", "")
    string = string.replace("\\;", "")
    string = string.replace("\\quad", "")

    # replace \\ with \
    # string = string.replace("\\\\", "\\")
    # string = string.replace("\\\\", "\\")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    ## fix for o1 and o3-mini: these models may use unicode characters for some operators.
    string = _fix_unicode_perturb(string)


    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    string = _fix_sqrt_perturb(string)
    string = _fix_tan_perturb(string)
    #string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs_perturb(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b_perturb(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string

def is_equiv_perturb(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string_perturb(str1)
        ss2 = strip_string_perturb(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2