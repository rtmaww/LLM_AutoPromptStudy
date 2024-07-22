from data.data_utils import subsample_data
import sys
from torch.quasirandom import SobolEngine
sys.path.append('AutoPrompter')
from LLM.llm_inference.APIInference import APIInference
from LLM.llm_inference.WhiteBoxInference import SoftPrompt_Inference
from utils import check_template_kwargs
from LLM.thread_utils import mutiThreadInference

'''
kwargs for initialization
{
""
}
'''

class Initializer(object):

    def __init__(self, inferencer=None) -> None:
        
        self.initial_prompt = ""
        self.inferencer = inferencer  
        
    
    def prompt_initialization(self,template=None,initial_kwargs=None):
        if '[DESC]' in initial_kwargs:
            self.initial_prompt = initial_kwargs['[DESC]']
        else:
            self.initial_prompt = ""
        if template is None:
            return self.initial_prompt
        else:
            check_template_kwargs(template=template,
                                  kwargs=initial_kwargs)
            llm_prompt = template.fill(initial_kwargs)
            print("initial instruction:\n")
            if isinstance(llm_prompt,list):
                for item in llm_prompt:
                    print(item)
            else:
                print(llm_prompt)
            return self.initialize_inference(llm_prompt)
            
    

    def initialize_inference(self, llm_prompt=""):
        if isinstance(self.inferencer,APIInference):
            hsitory,result = mutiThreadInference(
                prompts=llm_prompt,
                inferencer=self.inferencer,
                #gen_kwargs = self.gen_kwargs
            )
        elif isinstance(self.inferencer,SoftPrompt_Inference):
            self.inferencer.init_prompt = llm_prompt
            soft_embed = SobolEngine(dimension=self.inferencer.intrinsic_dim, scramble=True, seed=0).draw(self.gen_n)
            result = self.inferencer.generate_text(
                prompt=llm_prompt,
                soft_prompt_embd=soft_embed
                )
        else:
            history,result  = self.inferencer.generate_text(llm_prompt)
        
        
        if isinstance(result[0],list):
            return_result = []
            for item in result:
                return_result.extend(item)
        else:
            return_result = result
        return return_result
    
