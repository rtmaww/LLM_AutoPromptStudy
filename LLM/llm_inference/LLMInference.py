import torch
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer

from config_file import tkwargs

'''
kwargs for LLMInference:

GPT_Inference:

    ``
    Required:
        "model":['turbo','gpt4']
    Optional:
        "n":  default = 1
              count of return answer for a prompt in LLM query
    ``

Llama_Inference:

    ``
    Required:
        "model":['Llama_7B','Llama_13B','Llama_70B']
    ``
    

HFModelInference

    ``
    Required:
        "HF_dir" : a path can load the huggingface model weight
    ``

SoftPrompt_Inference

    ``
    Required:
        "n_prompt_tokens": the number of soft prompt tokens (set 3/5/10)
        "intrinsic_dim" : the intrinstic dim of soft prompt (set 10)
        "random_proj": ["uniform","normal"]
                    matrix used for proj tensor from ``intrinstic dim`` to  ``embedding dim`` 
    Optional Required:
        'alpha': Required if  ``random_proj==normal``
        'sigma': Required if  ``random_proj==normal``
    ``

    
Optional for all LLM Inferencer:

    "system_prompt" : default = "A chat between a curious user and an artificial intelligence assistant."\
                                " The assistant gives helpful, detailed, and polite answers to the user's questions."
    "top_p" : default = 0.9
    "temperature": default = 1.0
    "max_new_tokens": default = 400
    
'''

class LLMInference(object):

    def __init__(self,promptBuilder = None,gen_kwargs = {}) -> None:
        self.gen_kwargs = {
            "n" : 1,
            "system_prompt" : "A chat between a curious user and an artificial intelligence assistant."\
                                " The assistant gives helpful, detailed, and polite answers to the user's questions.",
            "temperature" : 1.0,
            "top_p" : 0.9,
            "max_new_tokens" : 1000
        }
        for k,v in gen_kwargs.items():
            if k in self.gen_kwargs:
                self.gen_kwargs[k] = v
        #self.set_gen_conf(gen_kwargs)
        self.system_prompt = self.gen_kwargs['system_prompt']
        self.gen_conf = GenerationConfig.from_dict(self.gen_kwargs)
        #if gen_kwargs:
        
        self.optional_params = ['n','system_prompt','temperature','top_p','max_new_tokens']

        print("Create Inferencer Object..\n")
        for param_name in self.optional_params:
            if param_name not in gen_kwargs:
                print(Warning(f"use default value of {param_name}:{self.gen_kwargs[param_name]},"\
                    f"you can set customized {param_name} in gen_kwargs in ``{param_name}`` ."))

        

        self.promptBuilder = promptBuilder
        
        self.model = AutoModelForCausalLM.from_pretrained(gen_kwargs['HF_dir']).to(**tkwargs) \
            if 'HF_dir' in gen_kwargs else None
        self.tokenizer = AutoTokenizer.from_pretrained(gen_kwargs['HF_dir'],padding_side = "left") \
            if 'HF_dir' in gen_kwargs else None
        
       
       
        
        
    
    def generate_text(self,prompts):
        pass

    




class LLMInferenceResult():

    def __init__(self,dialog_history,result) -> None:
        self.history = dialog_history
        if not isinstance(result,list):
            result = [result]
        self.result = result
