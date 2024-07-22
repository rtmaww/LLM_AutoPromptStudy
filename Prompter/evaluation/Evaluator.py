
import sys
sys.path.append("AutoPrompter")
from Prompter.utils import InsRecord
from LLM.thread_utils import mutiThreadInference

from LLM.llm_inference.APIInference import APIInference
import numpy as np
from Prompter.utils import check_template_kwargs
import re
import json

class Evaluator():
    def __init__(self,inference = None,scorer = None) -> None:
        self.scorer = scorer
        self.inferencer = inference
        # if inference is not None:
        #     self.inferencer.set_gen_conf(kwargs)
        
    
    def build_prompt(self,instruction,examples,evaluation_kwargs):
        #evaluation_kwargs = {}
        evaluation_kwargs['[INS]'] = instruction
        query_prompts = []
        if isinstance(examples[0],list):
            for ind in range(len(examples[0])):
                evaluation_kwargs['[EXAMPLE]'] = (examples[0][ind],examples[1][ind])
                query_prompts.append(self.template.fill(evaluation_kwargs))

        return query_prompts
    
    def evaluation_inference(self,llm_prompt):
        
        if isinstance(self.inferencer,APIInference):
            hsitory,result = mutiThreadInference(
                prompts=llm_prompt,
                inferencer=self.inferencer,
                #gen_kwargs = self.gen_kwargs
            )
        else:
            history,result  = self.inferencer.generate_text(llm_prompt)
       
        return result
    
    def extract_format_output(self,results):
        KEY_NAME = {'answer'}
        format_results = []
        format_error_number = 0
        # pattern = r'<OUTPUT>(.*?)<\/OUTPUT>'
        pattern = r'({.*?})'
        
        for result in results:
            matches = re.findall(pattern, result, re.DOTALL)
            format_error = True


        
       
        #result = ""
            if matches:
                for match in matches:
                    match = match.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("'", '"')
                    try:
                        match = json.loads(match)
                        for key in KEY_NAME:
                            if key in match:
                                result = str(match[key])
                                format_error = False
                                break
                              
                    except:
                        continue
            format_results.append(result)
            if format_error:
                format_error_number += 1
        print(f"extract format answer from {len(results)} outputs.\nformat error count:{format_error_number} ")  
        return format_results
       

    def evaluate(self,template,instructions,evaluation_kwargs_):
        check_template_kwargs(template=template,
                              kwargs=evaluation_kwargs_)
        if not isinstance(instructions,list):
            instructions = [instructions]
        evaluation_kwargs = evaluation_kwargs_.copy()
        self.template = template
        # if not isinstance(examples,list):
        #     examples = [examples]

        llm_prompts = []
        examples = evaluation_kwargs['[EXAMPLE]']
        
        for instruction in instructions:
            query_prompt = self.build_prompt(instruction,examples,evaluation_kwargs)
            if isinstance(query_prompt,list):
                llm_prompts.extend(query_prompt)
                print(query_prompt[0])
            else:
                llm_prompts.append(query_prompt)
                print(query_prompt)
       
        # for element in llm_prompts:
        #     print(element)
        result = self.evaluation_inference(llm_prompts)
        format_results = self.extract_format_output(result)
        trans_format_result = np.array(format_results).reshape(len(instructions),-1).tolist()
        filter_ind = []
        for result_ind,ins_result_list in enumerate(trans_format_result):
            result_dict = {}
            for item in ins_result_list:
                if item not in result_dict:
                    result_dict[item] = 0
                result_dict[item] += 1
            # for k,v in result_dict.items():
            #     if len(examples[0]) == 50 and v > 45:
            #         filter_ind.append(result_ind)
        
        inputs,correct_outputs = examples
        concat_correct_outputs = []
        for i in range (len(instructions)):
            concat_correct_outputs.extend(correct_outputs)
        score = self.scorer.score(predictions=format_results,
                                  ground_truths = concat_correct_outputs)
        
        result = np.array(result).reshape(len(instructions),-1).tolist()
        scores = np.array(score).reshape(len(instructions),-1).tolist()
        for filter_index in filter_ind:
            for i in range(len(scores[filter_index])):
                scores[filter_index][i] = -1*scores[filter_index][i]
            

        return EvaluatorResult(prompts=instructions,
                               scores=scores,
                               predictions=result)
                
        
class EvaluatorResult():
    def __init__(self,prompts=[],scores=[],predictions=[]) -> None:
        self.prompts = prompts
        self.scores = scores
        self.predictions = predictions
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            n = key.stop
            truncated_prompts = self.prompts[:n]
            truncated_scores = self.scores[:n]
            truncated_predictions = self.predictions[:n]
            return EvaluatorResult(truncated_prompts, truncated_scores, truncated_predictions)
        elif isinstance(key,int):
            truncated_prompts = self.prompts[key]
            truncated_scores = self.scores[key]
            truncated_predictions = self.predictions[key]
            return EvaluatorResult(truncated_prompts, truncated_scores, truncated_predictions)

        
    
    def add_item(self,item):
        self.prompts.extend(item.prompts)
        self.scores.extend(item.scores)
        self.predictions.extend(item.predictions)

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""
        if method == 'mean':
            return [np.mean(s) for s in self.scores]
        elif method == 'median':
            return [np.median(s) for s in self.scores]
        elif method == 'std':
            return [np.std(s) for s in self.scores]
        elif method == 'max':
            return [np.max(s) for s in self.scores]
        elif method == 'min':
            return [np.min(s) for s in self.scores]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.scores]
        else:
            raise ValueError('Invalid method: {}'.format(method))
    
    def sorted(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        # Sort prompts by score
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1],reverse=True)
        #sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts),reverse=True)]
        #sorted_scores = sorted(scores)
        sorted_indexes  = [index for index,score in sorted_indices]

        sorted_scores = [self.scores[index] for index in sorted_indexes]
        sorted_prompts = [self.prompts[index] for index in sorted_indexes]
        
        sorted_predictions = [self.predictions[index] for index in sorted_indexes]

        return EvaluatorResult(sorted_prompts,sorted_scores,sorted_predictions)

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        return self.prompts, scores
    