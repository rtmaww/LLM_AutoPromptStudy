import sys
sys.path.append("AutoPrompter/Prompter")
from Initializer import Initializer
import copy
from Selector import Topn_Selector
from evaluation.Evaluator import Evaluator
import os
from optimization import Optimizer
from LLM.llm_inference.APIInference import GPT_Inference,Llama_Inference
from LLM.llm_inference.HFModelInference import HFModelInference
from LLM.llm_inference.WhiteBoxInference import SoftPrompt_Inference
from evaluation.Scorer import Exact_Match_Scorer,Exact_Set_Scorer,Contains_Scorer  #F1_Scorer,
from template import template
from evaluation.Evaluator import EvaluatorResult
from data.data_process import data_prepare
from data.data_utils import get_task_desc,subsample_data,create_split
import argparse
import json
from utils import InsRecord,format_feedback,extract_instruction,extract_text_format
from args import parse_args
from conf import Initial_conf,merge_conf,evaluation_gen_conf,generation_gen_conf
import random

# random.seed(2)
def get_error_example(evaluation_result:EvaluatorResult,examples:list,error_demo_cnt:int=2):
    prompts,scores,predictions = evaluation_result.prompts,\
                                evaluation_result.scores,evaluation_result.predictions
   
    all_error_examples = []
    for score_list,prediction_list in zip(scores,predictions):
        error_answer_index = []
        for score_ind,score in enumerate(score_list):
            if score == 0:
                error_answer_index.append(score_ind)
        if len(error_answer_index) > 0:
            select_error_index = random.sample(error_answer_index,min(len(error_answer_index),error_demo_cnt))
        else:
            select_error_index = []
        error_examples = [[],[],[]]
        inputs,outputs = examples
        for index in select_error_index:
            error_examples[0].append(inputs[index])
            error_examples[1].append(outputs[index])
            error_examples[2].append(prediction_list[index])
        all_error_examples.append(error_examples)
    
    return all_error_examples
                
     
        
class AutoPrompter(object):

    def __init__(
            self,args
            ) -> None:
        
        
        self.eval_number = args.EAVL_DATA_NUMBER
        self.test_number = args.TEST_DATA_NUMBER
        self.initial_stylization= args.initial_stylization
        self.eval_model = args.eval_model
        self.Total_Evaluation_Results_Record = EvaluatorResult()
        self.Total_Test_Results_Record = EvaluatorResult()
        self.optimization_record = InsRecord()
        self.set_Constant()  
        
        self.get_Data(args)
        self.task = args.task
        self.task_cty = args.task_cty
        self.set_Conf()

        self.initial_inferencer = GPT_Inference(promptBuilder=None,
                                                gen_kwargs=self.initial_infer_conf)
        if self.eval_model == 'turbo':
            print('evaluation with GPT 3.5')
            self.eval_inferencer = GPT_Inference(promptBuilder=None,
                                               gen_kwargs=self.eval_infer_conf)
            self.ckpt_path = "AutoPrompter/35_instruction_ckpt"
        else:
            print('evaluation with Llama')
            self.eval_inferencer = Llama_Inference(promptBuilder=None,
                                               gen_kwargs=self.eval_infer_conf)
            self.ckpt_path = "AutoPrompter/Llama_instruction_ckpt"

        
        
        self. scorer = Exact_Match_Scorer()
        
        if args.initial_stylization:
            self.initial_template = template.Initial_Multi_Prompt_Desc_Fewshot_Template()
        else:
            self.initial_template = template.Initial_Prompt_Desc_Fewshot_Template()

        self.evaluation_template = template.EvaluationTemplate()

        

    def get_Data(self,args):
        self.prompt_gen_data,self.eval_data,self.test_data = data_prepare(args,prompt_gen_data=True)
    
    def set_Constant(self):
        #self.initial_stylization = args.initial
        if not self.initial_stylization:
            self.GEN_BATCH_SIZE = 5
            self.GEN_SUBSAMPLE = 2
            self.GEN_DATA_NUMBER = 4
            self.OPS_STEP = 10
        else:
            #generate with seed
            self.GEN_BATCH_SIZE = 2
            self.GEN_SUBSAMPLE = 5
            self.GEN_DATA_NUMBER = 4
            self.OPS_STEP = 10

        self.STORE_BATCH_SIZE = 10
        
        
        self.EAVL_DATA_NUMBER = self.eval_number
        self.TEST_DATA_NUMBER = self.test_number
    
    def set_Conf(self):
        self.eval_conf = {"[EXAMPLE]":self.eval_data}
        self.test_conf = {"[EXAMPLE]":self.test_data}
        self.task_desc = get_task_desc(self.task,self.task_cty)
        if self.initial_stylization:
            with open("AutoPrompter/Prompter/data/bigbench-ii/raw/ins.json",'r',encoding='utf-8') as f:
                initial_style = json.load(f)
            self.initial_conf = {
            "[DESC]":get_task_desc(self.task,self.task_cty),
            "[SEED_DICT]":initial_style
            }
        else:
            self.initial_conf = {
            "[DESC]":get_task_desc(self.task,self.task_cty)
            }
     
        eval_infer_conf = {
            "model" : self.eval_model,
            "system_prompt":"You are a helpful assistant."
        }
      
        initial_infer_conf = {
            "model" : "gpt-4",
            "n":self.GEN_BATCH_SIZE,
            "system_prompt":"You are a helpful assistant."
        }
        self.initial_infer_conf = merge_conf(generation_gen_conf,initial_infer_conf)
        self.eval_infer_conf = merge_conf(evaluation_gen_conf,eval_infer_conf)
    
    def evaluate_ins(self,ins_dict,record = False):
        self.evaluator = Evaluator(
            inference=self.eval_inferencer,
            scorer=self.scorer,
            )
        for step,ins_list in ins_dict.items():
            evaluation_result = self.evaluator.evaluate(
                template=self.evaluation_template,
                instructions=ins_list,
                evaluation_kwargs_=self.eval_conf
                )
            self.optimization_record.add_Ins_Record(
                step=step,
                Record=evaluation_result)
        
        if record:
            if self.eval_model == "turbo":
                prefix = "turbo_result"
            else:
                prefix = "recal_Llama_result"
            evaluation_file = "evaluation_style"  if self.initial_stylization else  "evaluation"
            
            with open(f'AutoPrompter/{prefix}/{self.method}_{self.task_cty}_{self.task}_{evaluation_file}.json','w',encoding='utf-8') as f:
                step_record = self.optimization_record.step_record
                eval_inputs,eval_outputs = self.eval_data
                for step,evaluation_results in step_record.items():
                    f.write(f'Optimization step{step}:\n\n')
                    predictions = evaluation_results.predictions
                    instructions,scalar_score = evaluation_results.in_place()
                    for ind,instruction in enumerate(instructions):
                        f.write(f'##INS\n{instruction}\t{scalar_score[ind]}\n')
                        cur_ins_prediction = predictions[ind]
                        for (eval_input,eval_output,prediction) in zip(eval_inputs,eval_outputs,cur_ins_prediction):
                            f.write(f'##Input:\n{eval_input}\n##Correct answer:\n{eval_output}\n##output:\n{prediction}\n')
                            
            with open(f'AutoPrompter/{prefix}/{self.method}_{self.task_cty}_{self.task}_record.json','w',encoding='utf-8') as f:
                step_record = self.optimization_record.step_record
                for step,evaluation_results in step_record.items():
                    f.write(f'Optimization step{step}:\n\n')
                    instructions,scalar_score = evaluation_results.in_place()
                    for ind,instruction in enumerate(instructions):
                        f.write(f'{instruction}\t{scalar_score[ind]}\n')
                        print(f'{instruction}\t{scalar_score[ind]}\n')
                  
            
    def test_ins(self,ins_list,step = 1, record = True):
        self.evaluator = Evaluator(
            inference=self.eval_inferencer,
            scorer=self.scorer,
            )
      
        test_result = self.evaluator.evaluate(
            template=self.evaluation_template,
            instructions=ins_list,
            evaluation_kwargs_=self.test_conf
        )
        
        self.optimization_record.add_Test_Record(
                step = step,
                Record = test_result
            )
        for step,test_result in self.optimization_record.step_test_record.items():
            prompts,scalar_scores = test_result.in_place()
            if record:
                for prompt,score in zip(prompts,scalar_scores):
                    print(f'{prompt}\t{score}')
                with open("test_ins_result.json","w",encoding='utf-8') as f:
                    for prompt,score in zip(prompts,scalar_scores):
                        f.write(f'{prompt}\t{score}\n')
        return scalar_scores
    
    def prompt_generation(self,ini_instructions = None,ini_step = 0,ini_from_start = False,load_from_ckpt=False):
        
       
        
        # Prompt initialization
        # 1. support read initial prompts in cache_file or parameter list.
        # 2. support initialized with LLM (need task examples or simple task desc.)
        if ini_instructions is None:
            instructions = []
            self.initializer = Initializer(inferencer=self.initial_inferencer)
            if self.initial_stylization:
                ini_instruction_path = f"AutoPrompter/Prompter/data/bigbench-ii/raw/style_{self.task}_initial_ins.json"
            else:
                ini_instruction_path = f"AutoPrompter/Prompter/data/bigbench-ii/raw/{self.task}_initial_ins.json"
            
            
            if not os.path.exists(ini_instruction_path):
                for cur_ind in range(self.GEN_SUBSAMPLE):
                    self.initial_conf['[EXAMPLE]'] = subsample_data(self.prompt_gen_data,self.GEN_DATA_NUMBER)
                    instructions.extend(self.initializer.prompt_initialization(template=self.initial_template,initial_kwargs=self.initial_conf))
                if args.initial_stylization:
                    for ind,item in enumerate(instructions):
                        instructions[ind] = instructions[ind].strip().lstrip("<START>").strip().rstrip("<END>").strip()
                        
                with open(ini_instruction_path,'w',encoding='utf-8') as f:
                    json.dump(instructions,f)
            else:
                with open(ini_instruction_path,'r',encoding='utf-8') as f:
                    instructions = json.load(f)
        else:
            print("initial instructions in parameters.")
            instructions = ini_instructions



        self.evaluator = Evaluator(
            inference=self.eval_inferencer,
            scorer=self.scorer,
            )
        self.optimizer = Optimizer.LLM_Optimizer(
            inferencer=self.gen_inferencer,
        ) 
        self.selector = Topn_Selector()

        best_ins_per_step = []
        ops_history = None
        
        ##instruction ckpt for chatGPT
        if ini_instructions is not None:
            self.ckpt_path = "continual_instruction_ckpt"
        if self.initial_stylization:
            self.ckpt_path = f"AutoPrompter/Multi_initial_{self.eval_model}_instruct_ckpt"
       

        for step in range(self.OPS_STEP):
          
            if step == 0:
                ckpt_path = f'{self.ckpt_path}/ini_{self.task}_ins.json'
            else:
                ckpt_path = f'{self.ckpt_path}/{self.method}_{self.task}_step{step}_ins.json'
            if os.path.exists(ckpt_path) and load_from_ckpt:
                with open(ckpt_path,'r',encoding='utf-8') as f:
                    ckpt_ins = json.load(f)
                evaluation_result = EvaluatorResult(prompts=ckpt_ins['prompts'],
                                                    scores=ckpt_ins['scores'],
                                                    predictions = ckpt_ins['predictions'])
            else:
                evaluation_result = self.evaluator.evaluate(
                    template=self.evaluation_template,
                    instructions=instructions,
                    evaluation_kwargs_=self.eval_conf
                    )
                if load_from_ckpt:
                    with open(ckpt_path,'w',encoding='utf-8') as f:
                        json.dump({"prompts":evaluation_result.prompts,"scores":evaluation_result.scores,"predictions":evaluation_result.predictions},f)
                
            
            self.Total_Evaluation_Results_Record.add_item(evaluation_result)
            

            sorted_evaluation_result = evaluation_result.sorted()
            best_ins_per_step.append(sorted_evaluation_result[0])
            cur_instructions,scalar_score = sorted_evaluation_result.in_place()
          
            print(f'step{step}:\n')
            for cur_ins,score in zip(cur_instructions,scalar_score):
                print(f'{cur_ins}\t{score}\n')
            
            
            record_list = sorted(scalar_score,reverse=True)
            avg_number = min(len(record_list),5)
            print(f'step{step}:\n')
            print (f'max_score:{max(scalar_score)}')
            print(f'avg_score:{round(sum(record_list[:avg_number])/avg_number,4)}')

            print(f'step{step}:\nBest instructions in dev:\n{cur_instructions[0]}\t{scalar_score[0]}\n')
            if step == 0:
                test_ckpt_path = f'{self.ckpt_path}/ini_{self.task}_test_ins.json'
            else:
                test_ckpt_path = f'{self.ckpt_path}/{self.method}_{self.task}_step{step}_test_ins.json'
            
      
            if os.path.exists(test_ckpt_path) and load_from_ckpt:
                with open(test_ckpt_path,'r',encoding='utf-8') as f:
                    ckpt_ins = json.load(f)
                test_result = EvaluatorResult(prompts=ckpt_ins['prompts'],
                                                    scores=ckpt_ins['scores'],
                                                    predictions = ckpt_ins['predictions'])
            else:
            
                test_result = self.evaluator.evaluate(
                    template=self.evaluation_template,
                    instructions=cur_instructions[0],
                    evaluation_kwargs_=self.test_conf
                    )
              
                if load_from_ckpt:
                    with open(test_ckpt_path,'w',encoding='utf-8') as f:
                        json.dump({"prompts":test_result.prompts,"scores":test_result.scores,"predictions":test_result.predictions},f)
                
                
            
       
            _,test_score = test_result.in_place()
            print(f'step{step}:\ntest_result:  {test_score}')
            self.optimization_record.add_Ins_Record(
                step=step,
                Record=evaluation_result)
            self.optimization_record.add_Ops_Record(
                step=step,
                dialog_history=ops_history
            )
            self.optimization_record.add_Test_Record(
                step = step,
                Record = test_result
            )
            
            
            
            check_path = f'{self.ckpt_path}/{self.method}_{self.task}_step{step+1}_ins.json'
            
            if step != self.OPS_STEP-1 :
                if (not os.path.exists(check_path)) or (not load_from_ckpt):
                    if  (step != 0) or (not ini_from_start):
                        filter_evaluation_result = self.selector.instruction_selection(evaluation_result,self.STORE_BATCH_SIZE)
                        filtered_instructions = filter_evaluation_result.prompts
                    else:
                        filter_evaluation_result = evaluation_result
                        filtered_instructions = evaluation_result.prompts
                    

                    ops_history,instructions = self.optimize(filter_evaluation_result)
                    
                    
        if ini_instructions is not None:
            ini_prefix = f"initial_step{ini_step}_"
        else:
            ini_prefix = ""
        prefix = ini_prefix + f"{self.eval_model}_result"
        evaluation_file = "evaluation_style"  if self.initial_stylization else  "evaluation"
        eval_record = "record_style"  if self.initial_stylization else  "record"
        optimization_record = "optimization_style"  if self.initial_stylization else  "optimization"
        test_file = "test_style" if self.initial_stylization else "test"
        test_record_file = "test_record_style"  if self.initial_stylization else  "test_record"
        
        if not os.path.exists(f'AutoPrompter/{prefix}'):
            os.makedirs(f'AutoPrompter/{prefix}')
        

        with open(f'AutoPrompter/{prefix}/{self.method}_{self.task_cty}_{self.task}_{evaluation_file}.json','w',encoding='utf-8') as f:

            step_record = self.optimization_record.step_record
            eval_inputs,eval_outputs = self.eval_data
            for step,evaluation_results in step_record.items():
                f.write(f'Optimization step{step}:\n\n')
                predictions = evaluation_results.predictions
                instructions,scalar_score = evaluation_results.in_place()
                for ind,instruction in enumerate(instructions):
                    f.write(f'##INS\n{instruction}\t{scalar_score[ind]}\n')
                    cur_ins_prediction = predictions[ind]
                    for (eval_input,eval_output,prediction) in zip(eval_inputs,eval_outputs,cur_ins_prediction):
                        f.write(f'##Input:\n{eval_input}\n##Correct answer:\n{eval_output}\n##output:\n{prediction}\n')
        
        
        
        with open(f'AutoPrompter/{prefix}/{self.method}_{self.task_cty}_{self.task}_{eval_record}.json','w',encoding='utf-8') as f:

            step_record = self.optimization_record.step_record
            for step,evaluation_results in step_record.items():
                f.write(f'\n\nOptimization step{step}:\n\n')
               
                instructions,scalar_score = evaluation_results.in_place()
                for instruction,score in zip(instructions,scalar_score):
                    f.write(f'{instruction}\t{score}\n')
        
        
        with open(f'AutoPrompter/{prefix}/{self.method}_{self.task_cty}_{self.task}_{optimization_record}.json','w',encoding='utf-8') as f:
            step_ops_record = self.optimization_record.step_ops_record
            json.dump(step_ops_record,f)


        test_record = []
        with open(f'AutoPrompter/{prefix}/{self.method}_{self.task_cty}_{self.task}_{test_file}.json','w',encoding='utf-8') as f:
            step_test_record = self.optimization_record.step_test_record
            test_inputs,test_outputs = self.test_data
            for step,test_results in step_test_record.items():
                f.write(f'Test step{step}:\n\n')
                predictions = test_results.predictions
                instructions,scalar_score = test_results.in_place()
                test_record.append(f'Instruction:\n{instructions[0]}\t{scalar_score[0]}\n')
                for ind,instruction in enumerate(instructions):
                    f.write(f'##INS\n{instruction}\t{scalar_score[ind]}\n')
                    cur_ins_prediction = predictions[ind]
                    for (test_input,test_output,prediction) in zip(test_inputs,test_outputs,cur_ins_prediction):
                        f.write(f'##Input:\n{test_input}\n##Correct answer:\n{test_output}\n##output:\n{prediction}\n')
        
        
        with open(f'AutoPrompter/{prefix}/{self.method}_{self.task_cty}_{self.task}_{test_record_file}.json','w',encoding='utf-8') as f:
            for item in test_record:
                f.write(item)

        return instructions


class APO_Prompter(AutoPrompter):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.method = args.ops_method
        self.STORE_BATCH_SIZE = args.STORE_BATCH_SIZE
        self.GEN_DATA_NUMBER = args.GEN_DATA_NUMBER
        self.OPS_STEP = args.OPS_STEP
        
        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
        #slots[INS | ERROR EXAMPLE]
        self.optimization_template_1 = template.Ops_Query_Feedback_Template()
        #slots [INS | EXAMPLE | FEEDBACK]
        self.optimization_template_2 = template.Ops_Update_Feedback_Template()


    def set_Conf(self):
        super().set_Conf()
       
        ops_infer_conf = {
            "model" : "gpt-4",
            "n":self.GEN_DATA_NUMBER ,
            "system_prompt":"You are a helpful assistant."
        }
        
        self.ops_infer_conf = merge_conf(generation_gen_conf,ops_infer_conf)
    

    
    def optimize(self,evaluation_result,error_example = None):

        instructions = evaluation_result.prompts
        if error_example == None:
            ERROR_EXAMPLE_COUNT = 4

            error_example = get_error_example(
                evaluation_result= evaluation_result,
                examples=self.eval_data,
                error_demo_cnt=ERROR_EXAMPLE_COUNT
            )
        optimized_instructions = []
        optimized_histories = [[] for i in range(len(instructions))]
        for ind,instruction in enumerate(instructions):

            optimize_kwargs = {
                "[ERROR_EXAMPLE]":error_example[ind],
                "[EXAMPLE]":error_example[ind][:2]
            }

            feedback_history,feedbacks = self.optimizer.prompt_optimize(
                template=self.optimization_template_1,
                prompts=instruction,
                optimize_kwargs=optimize_kwargs
                )
            feedbacks = format_feedback(feedbacks)
            optimize_kwargs['[FEEDBACK]'] = feedbacks
            update_history,instructions = self.optimizer.prompt_optimize(
                template=self.optimization_template_2,
                prompts=instruction,
                optimize_kwargs=optimize_kwargs
            )
            instructions = extract_instruction(instructions)
            optimized_instructions.extend(instructions)
            optimized_histories[ind].append([feedback_history[0],update_history[0]])
        return optimized_histories,optimized_instructions
        
class Random_Hint(AutoPrompter):
    
    def __init__(self,args) -> None:

        super().__init__(args)
        self.method = args.ops_method
        self.STORE_BATCH_SIZE = args.STORE_BATCH_SIZE
        self.GEN_DATA_NUMBER = args.GEN_DATA_NUMBER
        self.OPS_STEP = args.OPS_STEP
       
        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
       
        self.optimization_template = template.Ops_Error_Category_Template()
      

    def set_Conf(self):
        super().set_Conf()
       
        ops_infer_conf = {
            "model" : "gpt-4",
            "n":self.GEN_DATA_NUMBER ,
            "system_prompt":"You are a helpful assistant."
        }
        self.ops_infer_conf = merge_conf(generation_gen_conf,ops_infer_conf)
    

    
    def optimize(self,evaluation_result,error_example = None):

        instructions = evaluation_result.prompts
        if error_example == None:
            ERROR_EXAMPLE_COUNT = 4

            error_example = get_error_example(
                evaluation_result= evaluation_result,
                examples=self.eval_data,
                error_demo_cnt=ERROR_EXAMPLE_COUNT
            )
        optimized_instructions = []
        optimized_histories = []
        for ind,instruction in enumerate(instructions):

            optimize_kwargs = {
                "[ERROR_EXAMPLE]":error_example[ind],
                #"[EXAMPLE]":error_example[ind][:2]
            }

            history,instructions = self.optimizer.prompt_optimize(
                template=self.optimization_template,
                prompts=instruction,
                optimize_kwargs=optimize_kwargs
                )
            
            instructions = extract_instruction(instructions)
            optimized_instructions.extend(instructions)
            optimized_histories.append(history)
        return optimized_histories,optimized_instructions
        

class Muti_Initial_Prompter(AutoPrompter):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.method = args.ops_method
        self.STORE_BATCH_SIZE = args.STORE_BATCH_SIZE
        self.GEN_DATA_NUMBER = args.GEN_DATA_NUMBER
        self.OPS_STEP = args.OPS_STEP
        
        self.optimization_template = template.Initial_Multi_Prompt_Desc_Fewshot_Template()

        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
     
    def set_Constant(self):
        super().set_Constant()
        self.method = "Muti_Initial"
     
    def set_Conf(self):
        super().set_Conf()
        #prompt_gen_data,_ = create_split(self.prompt_gen_data,5)
       
        
        ops_infer_conf = {
            "model":"gpt-4",
            "n":self.GEN_DATA_NUMBER,
            "system_prompt":"You are a helpful assistant."
        }
       
        self.ops_infer_conf = merge_conf(generation_gen_conf,ops_infer_conf)

    
    
    def optimize(self,evaluation_result,error_example = None):
        instructions = evaluation_result.prompts
        optimized_instructions = []
        optimized_histories = []
        for ind in range(len(instructions)):

            optimize_kwargs = {
                "[EXAMPLE]":error_example[ind][:2],
                "[DESC]":self.task_desc
            }
            histories,new_instructions = self.optimizer.prompt_optimize(
            template=self.optimization_template,
            prompts=instructions[ind],
            optimize_kwargs=optimize_kwargs
            )
           
            optimized_instructions.extend(new_instructions)
            optimized_histories.append(histories)
        #optimize_kwargs = {}
        
        
        return optimized_histories,optimized_instructions



class APO_Agent_Prompter(AutoPrompter):

    def __init__(self,args) -> None:
        super().__init__(args)
        self.method = args.ops_method
        self.STORE_BATCH_SIZE = args.STORE_BATCH_SIZE
        self.GEN_DATA_NUMBER = args.GEN_DATA_NUMBER
        self.OPS_STEP = args.OPS_STEP
        
        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
        #slots[INS | ERROR EXAMPLE]
        self.optimization_template_1 = template.Ops_Query_Feedback_For_Agent()
        #slots [INS | EXAMPLE | FEEDBACK]
        self.optimization_template_2 = template.Ops_Update_For_Agent()

    
    def set_Conf(self):
        super().set_Conf()
       
        ops_infer_conf = {
            "model" : "gpt-4",
            "n":self.GEN_DATA_NUMBER ,
            "system_prompt":"You are a helpful assistant."
        }
        
        self.ops_infer_conf = merge_conf(generation_gen_conf,ops_infer_conf)
    

   
    def optimize(self,evaluation_result,error_example = None):

        instructions = evaluation_result.prompts
        if error_example == None:
            ERROR_EXAMPLE_COUNT = 4

            error_example = get_error_example(
                evaluation_result= evaluation_result,
                examples=self.eval_data,
                error_demo_cnt=ERROR_EXAMPLE_COUNT
            )
        optimized_instructions = []
        optimized_histories = [[] for i in range(len(instructions))]
        for ind,instruction in enumerate(instructions):

            optimize_kwargs = {
                "[ERROR_EXAMPLE]":error_example[ind],
            }

            feedback_history,feedbacks = self.optimizer.prompt_optimize(
                template=self.optimization_template_1,
                prompts=instruction,
                optimize_kwargs=optimize_kwargs
                )
            #feedbacks = format_feedback(feedbacks)
            optimize_kwargs['[FEEDBACK]'] = feedbacks[0]
            update_history,instructions = self.optimizer.prompt_optimize(
                template=self.optimization_template_2,
                prompts=instruction,
                optimize_kwargs=optimize_kwargs
            )
            instructions = extract_instruction(instructions)
            optimized_instructions.extend(instructions)
            optimized_histories[ind].append([feedback_history[0],update_history[0]])
        return optimized_histories,optimized_instructions
 



class APE_plus_Prompter(AutoPrompter):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.method = args.ops_method
        self.STORE_BATCH_SIZE = args.STORE_BATCH_SIZE
        self.GEN_DATA_NUMBER = args.GEN_DATA_NUMBER
        self.OPS_STEP = args.OPS_STEP
        
        self.optimization_template = template.Ops_Paraphrase_Plus_Template()

        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
     
     
    def set_Conf(self):
        super().set_Conf()
       
        ops_infer_conf = {
            "model":"gpt-4",
            "n":self.GEN_DATA_NUMBER,
            "system_prompt":"You are a helpful assistant."
        }
       
        self.ops_infer_conf = merge_conf(generation_gen_conf,ops_infer_conf)

   
    
    def optimize(self,evaluation_result,error_example = None):
        instructions = evaluation_result.prompts
        if error_example == None:
            ERROR_EXAMPLE_COUNT = 4

            error_example = get_error_example(
                evaluation_result= evaluation_result,
                examples=self.eval_data,
                error_demo_cnt=ERROR_EXAMPLE_COUNT
            )
        optimized_instructions = []
        optimized_histories = []
        for ind,instruction in enumerate(instructions):

            optimize_kwargs = {
                #"[ERROR_EXAMPLE]":error_example[ind],
                "[EXAMPLE]":error_example[ind][:2]
            }
            histories,instructions = self.optimizer.prompt_optimize(
            template=self.optimization_template,
            prompts=instruction,
            optimize_kwargs=optimize_kwargs
            )
           
            optimized_instructions.extend(instructions)
            optimized_histories.append(histories)
        #optimize_kwargs = {}
        
        
        return optimized_histories,optimized_instructions


class LLM_AS_OPtimizer_prompter(AutoPrompter):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.method = args.ops_method
        self.STORE_BATCH_SIZE = args.STORE_BATCH_SIZE
        self.GEN_DATA_NUMBER = args.GEN_DATA_NUMBER
        self.OPS_STEP = args.OPS_STEP
        
        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
        
        
        ##[EXAMPLE,INS_SCORE]
        self.optimization_template = template.Ops_Prompt_Score_Example_Template()
        
    def set_Constant(self):
        super().set_Constant()
        self.INS_SCORE_COUNT = 20
        # self.GEN_SUBSAMPLE = 1
      

    def set_Conf(self):
        super().set_Conf()
        ops_infer_conf = {
            "model" : "gpt-4",
            "n" : self.GEN_DATA_NUMBER,
            "system_prompt": "You are a helpful assistant."
        }
        #self.initial_infer_conf = merge_conf(generation_gen_conf,initial_infer_conf)
        self.ops_infer_conf = merge_conf(generation_gen_conf,ops_infer_conf)

    def optimize(self,evaluation_result,error_example = None,step = 2):
        instructions = evaluation_result.prompts
        if error_example == None:
            EXAMPLE_COUNT = 4
            error_example = get_error_example(
                evaluation_result= evaluation_result,
                examples=self.eval_data,
                error_demo_cnt=EXAMPLE_COUNT
            )

        prompts,scalar_scores = self.Total_Evaluation_Results_Record.sorted()[:self.INS_SCORE_COUNT].in_place()
        ins_score_dict = {}
        # for prompt,score in zip(prompts,scalar_scores):
        #     ins_score_dict[prompt] = score
        ind = len(prompts)-1
        while ind >=0:
            ins_score_dict[prompts[ind]] = scalar_scores[ind]
            ind -= 1
        optimized_instructions = []
        optimized_histories = []
        if step == 0:
            ops_count = int(20/self.OPS_BATCH)
        else:
            ops_count = int(10/self.OPS_BATCH)
            
        for ind in range(ops_count):
            optimize_kwargs = {
                "[INS_SCORE]":ins_score_dict,
                "[EXAMPLE]":error_example[ind][:2]
            }

            history,instructions = self.optimizer.prompt_optimize(
                template=self.optimization_template,
                prompts=None,
                optimize_kwargs=optimize_kwargs
                )
            extract_pattern = r'<INS>(.*?)</INS>'
            instructions = extract_text_format(instructions,pattern=extract_pattern)
            optimized_histories.append(history)
            optimized_instructions.extend(instructions)
        return optimized_histories,optimized_instructions
        
class APE_Prompter(AutoPrompter):
  
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.method = args.ops_method
        self.STORE_BATCH_SIZE = args.STORE_BATCH_SIZE
        self.GEN_DATA_NUMBER = args.GEN_DATA_NUMBER
        self.OPS_STEP = args.OPS_STEP
        
        self.optimization_template = template.Ops_ParaphraseTemplate()

        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
     
    
    
    def set_Conf(self):
        super().set_Conf()
        #prompt_gen_data,_ = create_split(self.prompt_gen_data,5)
       
        
        ops_infer_conf = {
            "model":"turbo",
            "n":self.GEN_DATA_NUMBER,
            "system_prompt":"You are a helpful assistant."
        }
       
        self.ops_infer_conf = merge_conf(generation_gen_conf,ops_infer_conf)

   
    def optimize(self,evaluation_result,error_example = None):
        optimize_kwargs = {}
        
        histories,instructions = self.optimizer.prompt_optimize(
            template=self.optimization_template,
            prompts=evaluation_result.prompts,
            optimize_kwargs=optimize_kwargs
            )
        return histories,instructions
   




if __name__ == "__main__":
    
    
    args = parse_args()
   
    if args.ops_method == "APO":
        prompter = APO_Prompter(args)
    elif args.ops_method == "APE":
        prompter = APE_Prompter(args)
    elif args.ops_method == "APO_Sum":
        prompter = Random_Hint(args)
    elif args.ops_method == "ORPO":
        prompter = LLM_AS_OPtimizer_prompter(args)
    elif args.ops_method == "APO_AGENT":
        prompter = APO_Agent_Prompter(args)
    else:
        raise ValueError ("not implemented")
    #prompter = Muti_Initial_Prompter(args)
    
    # ini_from_start: if excuted from step 0, you can set “ini_from_start=False” to allow start from middle step
    # ini_instructions: you can provide prompt list as initial prompt instead of using LLM for initialization.
    # load_from_ckpt: to save api , you can set  “load_from_ckpt = TRUE” to load intruction ckpt from step k 
    prompter.prompt_generation(ini_instructions = None,
                               ini_step = 0,
                               ini_from_start = False,
                               load_from_ckpt= False)  
