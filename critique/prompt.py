# Templates -------------------------------------------------------------------
# StrategyQA / yes-no
YES_NO_VANILLA_UNC = """Answer the following yes/no question and provide your uncertainty score. Your response should end with 'The answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is yes or no, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {}
"""

YES_NO_VANILLA_CONF = """Answer the following yes/no question and provide your confidence score. Your response should end with 'The answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is yes or no, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {}
"""

YES_NO_SELF_CRITIQUE_UNC = """You previously answered the following yes/no question with the response ended with: 'The answer is [initial_answer], and the uncertainty is [initial_uncertainty]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your uncertainty score. Answer the question and provide a new uncertainty score. Your response should end with 'The refined answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is yes or no, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure about the question, you should give a higher uncertainty percentage. 

Question: {}
Initial response: {}
"""

YES_NO_SELF_CRITIQUE_CONF = """You previously answered the following yes/no question with the response ended with: 'The answer is [initial_answer], and the confidence is [initial_confidence]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your confidence score. Answer the question and provide a new confidence score. Your response should end with 'The refined answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is yes or no, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your refined answer. If you are not sure about your refined answer, you should give a lower confidence percentage. 

Question: {}
Initial response: {}
"""

# Trivia / Hotpot open_end QA
OPEN_VANILLA_UNC = """Answer the following question and provide your uncertainty score. Your response should end with 'The answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is the direct answer to the question, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {}
"""
OPEN_VANILLA_CONF = """Answer the following question and provide your confidence score. Your response should end with 'The answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is the direct answer to the question, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {}
"""

OPEN_SELF_CRITIQUE_UNC = """You previously answered the following question with the response ended with: 'The answer is [initial_answer], and the uncertainty is [initial_uncertainty]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your uncertainty score. Answer the question and provide a new uncertainty score. Your response should end with 'The refined answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is the direct answer to the question, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure about the question, you should give a higher uncertainty percentage. 

Question: {}
Initial response: {}
"""

OPEN_SELF_CRITIQUE_CONF = """You previously answered the following question with the response ended with: 'The answer is [initial_answer], and the confidence is [initial_confidence]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your confidence score. Answer the question and provide a new confidence score. Your response should end with 'The refined answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is the direct answer to the question, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your refined answer. If you are not sure about your refined answer, you should give a lower confidence percentage. 

Question: {}
Initial response: {}
"""

# Multiple-choice (comparisonqa)
MC_VANILLA_UNC = """Answer the following multiple choice question. Select only one correct answer from the choices and give your uncertainty score about this question. Your response should end with 'The answer is [option_letter], and the uncertainty is [uncertainty_percentage]%' where the [option_letter] is one of A, B, C and D, and the uncertainty percentage should be a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure, you should give a higher uncertainty percentage.
Question: {} A. {}. B. {}. C. {}. D. {}
"""
MC_VANILLA_CONF = """Answer the following multiple choice question. Select only one correct answer from the choices and give your confidence score about this question. Your response should end with 'The answer is [option_letter], and the confidence is [confidence_percentage]%' where the [option_letter] is one of A, B, C and D, and the confidence percentage should be a number between 0 and 100, indicating how sure you are about your answer. If you are not sure, you should give a lower confidence percentage.
Question: {} A. {}. B. {}. C. {}. D. {}
"""

MC_SELF_CRITIQUE_UNC = """You previously answered the following multiple choice question with the response ended with: 'The answer is [initial_answer], and the uncertainty is [initial_uncertainty]%'. Now, refine your answer by reassessing the question, options, and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your uncertainty score. Select only one correct answer from the choices and provide a new uncertainty score. Your response should end with 'The refined answer is [option_letter], and the uncertainty is [uncertainty_percentage]%' where [option_letter] is one of A, B, C, or D, and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure about the question, you should give a higher uncertainty percentage. 

Question: {} A. {}. B. {}. C. {}. D. {}.
Initial response: {}
"""

MC_SELF_CRITIQUE_CONF = """You previously answered the following multiple choice question with the response ended with: 'The answer is [initial_answer], and the confidence is [initial_confidence]%'. Now, refine your answer by reassessing the question, options, and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your confidence score. Select only one correct answer from the choices and provide a new confidence score. Your response should end with 'The refined answer is [option_letter], and the confidence is [confidence_percentage]%' where [option_letter] is one of A, B, C, or D, and the confidence percentage is a number between 0 and 100, indicating how sure you are about your refined answer. If you are not sure about your refined answer, you should give a lower confidence percentage. 

Question: {} A. {}. B. {}. C. {}. D. {}.
Initial response: {}
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

MATH_SELF_CRITIQUE_UNC = """You previously answered the following math question and provided your uncertainty score based on these requirements:
1. Step-by-step reasoning: Provide a clear, step-by-step derivation of the answer, showing all necessary calculations and logical steps in the solution.
2. Final answer: Present the final answer [your_answer] in LaTeX format, wrapped in '\\boxed{{}}', ensuring the answer is concise and follows mathematical notation standards.
3. Uncertainty level: After the final answer, state your uncertainty in the solution as a percentage. If you are not sure, you should give a higher uncertainty percentage.
4. Concise language: Avoid redundancy and focus directly on the problem's core.
5. Format consistency: Ensure the answer format matches the problem's requirements (e.g., fractions, radicals, intervals).

Your initial response ended with: 'The answer is [initial_answer], and the uncertainty is [initial_uncertainty]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your uncertainty score. Answer the question and provide a new uncertainty score. Your response should end with 'The refined answer is [your_answer], and the uncertainty is [uncertainty_percentage]%' where [your_answer] is the final answer in LaTeX format, wrapped in '\\boxed{{}}', and the uncertainty percentage is a number between 0 and 100, indicating how uncertain you are about the question. If you are not sure about the question, you should give a higher uncertainty percentage. 

Question: {}
Initial response: {}
"""

MATH_SELF_CRITIQUE_CONF = """You previously answered the following math question and provided your confidence score based on these requirements:
1. Step-by-step reasoning: Provide a clear, step-by-step derivation of the answer, showing all necessary calculations and logical steps in the solution.
2. Final answer: Present the final answer [your_answer] in LaTeX format, wrapped in '\\boxed{{}}', ensuring the answer is concise and follows mathematical notation standards.
3. Confidence level: After the final answer, state your confidence in the solution as a percentage. If you are not sure, you should give a lower confidence percentage.
4. Concise language: Avoid redundancy and focus directly on the problem's core.
5. Format consistency: Ensure the answer format matches the problem's requirements (e.g., fractions, radicals, intervals).

Your initial response ended with: 'The answer is [initial_answer], and the confidence is [initial_confidence]%'. Now, refine your answer by reassessing the question and your initial reasoning. Consider any potential ambiguities or logical steps that could improve the accuracy of your response and the calibration of your confidence score. Answer the question and provide a new confidence score. Your response should end with 'The refined answer is [your_answer], and the confidence is [confidence_percentage]%' where [your_answer] is the final answer in LaTeX format, wrapped in '\\boxed{{}}', and the confidence percentage is a number between 0 and 100, indicating how sure you are about your refined answer. If you are not sure about your refined answer, you should give a lower confidence percentage. 

Question: {}
Initial response: {}
"""


GEN_CRIT_UNC_STRATEGYQA = """Uncertainty indicates how uncertain the student is about the question. If he is not sure, he should give a higher uncertainty percentage. You are a teacher expert in uncertainty calibration. A student previously answered a question and provided his uncertainty score. Please evaluate the calibration of his uncertainty score for the question based on his response. If his response is incorrect, the uncertainty percentage should be high. 

Question: {}
Correct Answer: {}
Facts: {}
Student's Response: {}

Using the facts and the correct answer as a reference, assess whether the uncertainty percentage in the student's response is well-calibrated, considering the clarity and strength of the reasoning provided and your own knowledge of the question. Is the uncertainty percentage appropriate, too high, or too low? Provide a brief explanation of your evaluation, focusing on how well his uncertainty aligns with the strength of his reasoning and the context of the question.
"""

GEN_CRIT_CONF_STRATEGYQA = """Confidence indicates how how sure the student is about his answer. If he is not sure, he should give a lower confidence percentage. You are a teacher expert in confidence calibration. A student previously answered a question and provided his confidence score. Please evaluate the calibration of his confidence score for the question based on his response. If his response is incorrect, the confidence percentage should be low. 

Question: {}
Correct Answer: {}
Facts: {}
Student's Response: {}

Using the facts and the correct answer as a reference, assess whether the confidence percentage in the student's response is well-calibrated, considering the clarity and strength of the reasoning provided and your own knowledge of the question. Is the confidence percentage appropriate, too high, or too low? Provide a brief explanation of your evaluation, focusing on how well his confidence aligns with the strength of his reasoning and the context of the question.
"""


GEN_CRIT_UNC_STRATEGYQA_THINK = GEN_CRIT_UNC_STRATEGYQA + "\nUse \"</think>\\n\\n\" to separate your thinking process and your final response."
GEN_CRIT_CONF_STRATEGYQA_THINK = GEN_CRIT_CONF_STRATEGYQA + "\nUse \"</think>\\n\\n\" to separate your thinking process and your final response."

GEN_CRIT_UNC_COMPARISONQA = """Uncertainty indicates how uncertain the student is about the question. If he is not sure, he should give a higher uncertainty percentage. You are a teacher expert in uncertainty calibration. A student previously answered a question and provided his uncertainty score. Please evaluate the calibration of his uncertainty score for the question based on his response. If his response is incorrect, the uncertainty percentage should be high. 

Question: {}  A. {}  B. {}  C. {}  D. {}
Correct Answer: {}
Student's Response: {}

Using the correct answer as a reference, assess whether the uncertainty percentage in the student's response is well-calibrated, considering the clarity and strength of the reasoning provided and your own knowledge of the question. Is the uncertainty percentage appropriate, too high, or too low? Provide a brief explanation of your evaluation, focusing on how well his uncertainty aligns with the strength of his reasoning and the context of the question.
"""


GEN_CRIT_CONF_COMPARISONQA = """Confidence indicates how how sure the student is about his answer. If he is not sure, he should give a lower confidence percentage. You are a teacher expert in confidence calibration. A student previously answered a question and provided his confidence score. Please evaluate the calibration of his confidence score for the question based on his response. If his response is incorrect, the confidence percentage should be low. 

Question: {}  A. {}  B. {}  C. {}  D. {}
Correct Answer: {}
Student's Response: {}

Using the correct answer as a reference, assess whether the confidence percentage in the student's response is well-calibrated, considering the clarity and strength of the reasoning provided and your own knowledge of the question. Is the confidence percentage appropriate, too high, or too low? Provide a brief explanation of your evaluation, focusing on how well his confidence aligns with the strength of his reasoning and the context of the question.
"""

GEN_CRIT_UNC_COMPARISONQA_THINK = GEN_CRIT_UNC_COMPARISONQA + "\nUse \"</think>\\n\\n\" to separate your thinking process and your final response."
GEN_CRIT_CONF_COMPARISONQA_THINK = GEN_CRIT_CONF_COMPARISONQA + "\nUse \"</think>\\n\\n\" to separate your thinking process and your final response."

GEN_CRIT_UNC_MATH = """Uncertainty indicates how uncertain the student is about the question. If he is not sure, he should give a higher uncertainty percentage. You are a teacher expert in uncertainty calibration. A student previously answered a question and provided his uncertainty score. Please evaluate the calibration of his uncertainty score for the question based on his response. If his response is incorrect, the uncertainty percentage should be high. 

Question: {}
Correct Answer: {}
Student's Response: {}

Using the correct answer as a reference, assess whether the uncertainty percentage in the student's response is well-calibrated, considering the clarity and strength of the reasoning provided and your own knowledge of the question. Is the uncertainty percentage appropriate, too high, or too low? Provide a brief explanation of your evaluation, focusing on how well his uncertainty aligns with the strength of his reasoning and the context of the question.
"""


GEN_CRIT_CONF_MATH = """Confidence indicates how how sure the student is about his answer. If he is not sure, he should give a lower confidence percentage. You are a teacher expert in confidence calibration. A student previously answered a question and provided his confidence score. Please evaluate the calibration of his confidence score for the question based on his response. If his response is incorrect, the confidence percentage should be low. 

Question: {}
Correct Answer: {}
Student's Response: {}

Using the correct answer as a reference, assess whether the confidence percentage in the student's response is well-calibrated, considering the clarity and strength of the reasoning provided and your own knowledge of the question. Is the confidence percentage appropriate, too high, or too low? Provide a brief explanation of your evaluation, focusing on how well his confidence aligns with the strength of his reasoning and the context of the question.
"""

GEN_CRIT_UNC_MATH_THINK = GEN_CRIT_UNC_MATH + "\nUse \"</think>\\n\\n\" to separate your thinking process and your final response."
GEN_CRIT_CONF_MATH_THINK = GEN_CRIT_CONF_MATH + "\nUse \"</think>\\n\\n\" to separate your thinking process and your final response."
