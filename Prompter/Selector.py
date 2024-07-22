from utils import InsRecord
import random
from evaluation.Evaluator import EvaluatorResult
class Topn_Selector(object):
    def __init__(self) -> None:
        pass
      
    def instruction_selection(self,evaluation_result,n):
         
        sorted_evaluation_result = evaluation_result.sorted()
        prompts, scalar_score = sorted_evaluation_result.in_place()
        start_point = 0
        cur_best_score = scalar_score[0]
        optional_index = []
        must_index = []
        while start_point < n:
            must_index.extend(optional_index)
            optional_index = []
            while start_point < len(prompts):
                score_differ = cur_best_score-scalar_score[start_point]
                if score_differ > 0.00005:
                    cur_best_score = scalar_score[start_point]
                  
                    
                    break
                else:
                    optional_index.append(start_point)
                    start_point += 1
            
        assert (len(must_index)+len(optional_index))>=n
        optional_count = n-len(must_index)
        selected_optional_index = random.sample(optional_index,optional_count)
        
        must_index.extend(selected_optional_index)
        build_prompts,build_scores,build_preds = [],[],[]
       
        for index in must_index:
            build_prompts.append(prompts[index])
            build_scores.append(sorted_evaluation_result.scores[index])
            build_preds.append(sorted_evaluation_result.predictions[index])
        selected_evaluation_result = EvaluatorResult(prompts=build_prompts,
                                                         scores=build_scores,
                                                         predictions=build_preds)
        
        
        
            
        return selected_evaluation_result
      
