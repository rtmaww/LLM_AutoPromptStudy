from typing import Any
from data.data_utils import subsample_data
import sys
from torch.quasirandom import SobolEngine
from Prompter.utils import check_template_kwargs
sys.path.append('AutoPrompter')
from LLM.llm_inference.APIInference import APIInference
from LLM.llm_inference.WhiteBoxInference import SoftPrompt_Inference
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
import torch
import time
from botorch import fit_gpytorch_model
from optimization.kernel import CombinedStringKernel,cma_es_concat
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.analytic import ExpectedImprovement
from config_file import tkwargs
from LLM.thread_utils import mutiThreadInference
import numpy as np
from config_file import SOFT_BATCH_SIZE
'''
Params for optimizer:

__init__(kwargs): 

Params in kwargs:

    ``
    Optional:
        system_prompt
        params for LLM Generation,eg. ``temperature`` ``max_new_tokens``
    ``

prompt optimize(optimize_kwargs):

Params in optimize_kwargs:

    INS_Optimizer : None

    INS_ErrorExample_Optimizer:
        Required:


'''
class Optimizer(object):

    def __init__(self, inferencer=None):#, kwargs=None) -> None:
        #self.gen_kwargs = kwargs
       
        self.inferencer = inferencer    
        # if self.inferencer is not None:
        #     self.inferencer.set_gen_conf(self.gen_kwargs) 


    def prompt_optimize(
            self,
            prompts=None,
            template=None,
            optimize_kwargs={}
         ):
       pass
    
class LLM_Optimizer(Optimizer):
    def __init__(self, inferencer=None) -> None:
        super().__init__(inferencer)
    def prompt_optimize(self, prompts=None, template=None, optimize_kwargs={}):
        check_template_kwargs(template,optimize_kwargs)
        optimize_kwargs_ = optimize_kwargs.copy()

        if isinstance(prompts,list):
            llm_prompt = []
            for prompt in prompts:
                optimize_kwargs_['[INS]'] = prompt
                llm_prompt.append(template.fill(optimize_kwargs_))
        else:
            if prompts:
                optimize_kwargs_['[INS]'] = prompts
            llm_prompt = template.fill(optimize_kwargs_)

        history = None
        if isinstance(llm_prompt,list):
            print(llm_prompt[0])
        else:
            print(llm_prompt)
        if isinstance(self.inferencer,APIInference):
            history,result = mutiThreadInference(
                prompts=llm_prompt,
                inferencer=self.inferencer,
                #gen_kwargs = self.gen_kwargs
            )
        elif isinstance(self.inferencer,SoftPrompt_Inference):
            result = self.inferencer.generate_text(
                prompt=None,
                soft_prompt_embd=llm_prompt
                )
        else:
            history,result  = self.inferencer.generate_text(llm_prompt)
        
       
        return history,result
  