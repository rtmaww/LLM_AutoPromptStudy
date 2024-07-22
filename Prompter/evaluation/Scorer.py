import numpy as np
import sys
sys.path.append("AutoPrompter/Prompter")
from abc import ABC,abstractmethod
from evaluation.utils import normalize_prediction
import re
# import Counter
 
class Scorer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def score(self,predictions,ground_truths):
        pass
    
class Exact_Match_Scorer(Scorer):
    
    def __init__(self) -> None:
        super().__init__()

    def score(self,predictions, ground_truths):
        scores = []
        count_label = {}
        format_answer = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            prediction_tokens = normalize_prediction(prediction, lowercase=True).split()
            format_answer.append(prediction_tokens)
            # if prediction_tokens[0] not in count_label:
            #     count_label[prediction_tokens[0]] = 0
            
            # count_label[prediction_tokens[0]] += 1
            
            if isinstance(ground_truth,list):
                exact = 0
                for item in ground_truth:
                    ground_truth_token =  normalize_prediction(item, lowercase=True).split()
                    if ground_truth_token == prediction_tokens:
                        exact = 1
                        break
                scores.append(exact)
            else:
            
                ground_truth_tokens = normalize_prediction(ground_truth, lowercase=True).split()
                scores.append(prediction_tokens==ground_truth_tokens)
        # for k,v in count_label:
        #     if v > 45:
        #         scores = [-1 for i in range(predictions)]
        return scores

class Exact_Set_Scorer(Scorer):
    
    def __init__(self) -> None:
        super().__init__()
    
    def score(self,predictions, ground_truths):
        scores = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            prediction_tokens = normalize_prediction(prediction, lowercase=True).split()
            if isinstance(ground_truth,list):
                exact = 0
                for item in ground_truth:
                    ground_truth_token =  normalize_prediction(item, lowercase=True).split()
                    temp_score = int(set(prediction_tokens) == set(ground_truth_token))
                    if temp_score > exact:
                        exact = temp_score
                scores.append(exact)
            else:
                ground_truth_tokens = normalize_prediction(ground_truth, lowercase=True).split()
                scores.append(int(set(prediction_tokens) == set(ground_truth_tokens)))

        return scores

class Contains_Scorer(Scorer):
    
    def __init__(self) -> None:
        super().__init__()
    
    def score(self,predictions, ground_truths):
        scores = []
        count_label = {}
        for prediction, ground_truth in zip(predictions, ground_truths):
            prediction_tokens = normalize_prediction(prediction, lowercase=True).split()
            if prediction_tokens not in count_label:
                count_label[prediction_tokens] = 0
            
            count_label[prediction_tokens] += 1
            
            if isinstance(ground_truth,list):
                exact = 0
                for item in ground_truth:
                    ground_truth_token =  normalize_prediction(item, lowercase=True).split()
                    if re.search(r'\b({0})\b'.format(prediction_tokens), ground_truth_token):
                        exact = 1
                        break
                scores.append(exact)
            else:
                ground_truth_tokens = normalize_prediction(ground_truth, lowercase=True).split()
                if re.search(r'\b({0})\b'.format(prediction_tokens), ground_truth_tokens):
                    scores.append(1)
                else:
                    scores.append(0)
        for k,v in count_label:
            if v > 45:
                scores = [-1 for i in range(predictions)]
            
    