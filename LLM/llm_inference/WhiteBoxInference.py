
import sys
sys.path.append("/Users/wangxiaolei/Downloads/AutoPrompter/LLM")
from llm_inference.LLMInference import LLMInference,LLMInferenceResult


import torch
import numpy as np




class WhiteBoxInference(LLMInference):

    def __init__(self, promptBuilder=None, gen_kwargs={}) -> None:

        assert 'HF_dir' in gen_kwargs,\
            ValueError('loading HFModel need a path which save the weight from huggingface.')
        
        assert promptBuilder is not None,\
            ValueError('HFmodel inference need a promptBuilder.')
        
        
        super().__init__(promptBuilder, gen_kwargs)
        #self.gen_conf = GenerationConfig.from_dict(self.gen_conf)
    
    # def set_gen_conf(self, gen_kwargs):
    #     return super().set_gen_conf(gen_kwargs)

    def generate_text(prompts):
        
        pass



class SoftPrompt_Inference(WhiteBoxInference):

    def __init__(self, promptBuilder=None, gen_kwargs={}) -> None:
        
        super().__init__(promptBuilder, gen_kwargs)

        required_params = ["n_prompt_tokens", "intrinsic_dim", "random_proj"]
        for param in required_params:
            assert param in gen_kwargs, \
            ValueError(f"SoftPrompt Inference needs the parameter: ``{param}`` in gen_kwargs.")
        
        self.n_prompt_tokens = gen_kwargs['n_prompt_tokens']
        self.intrinstic_dim = gen_kwargs['intrinsic_dim']
        self.random_proj = gen_kwargs['random_proj']
        

        self.embedding = self.model.get_input_embeddings().weight.clone()
        self.hidden_size = self.embedding.shape[-1]
        self.init_prompt = ""


        self.linear = None
        if self.intrinstic_dim != self.hidden_size:
            self.linear = torch.nn.Linear(self.intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)

            if self.random_proj == 'normal':

                mu_hat = self.embedding.reshape(-1).mean().item()
                std_hat = self.embedding.reshape(-1).std().item()
                mu = 0.0
                std = gen_kwargs['alpha'] * std_hat / (np.sqrt(self.intrinsic_dim) * gen_kwargs['sigma'])

                print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
                torch.nn.init.normal_(self.linear.weight, -1, 1)

            elif self.random_proj == 'uniform':  
                torch.nn.init.uniform_(self.linear.weight, -1, 1)
            



        



    # def set_gen_conf(self, gen_kwargs):
    #     return super().set_gen_conf(gen_kwargs)



    def generate_text(self, prompt=None,soft_prompt_embd=None):
        #prompts = super().generate_text(prompts)
        if prompt is None:
            prompt = self.init_prompt
        assert soft_prompt_embd.shape[-1] == self.intrinstic_dim,\
            ValueError(f"prompt dim need to equal intrintic dim:{self.intrinstic_dim}")

        if isinstance(prompt,list):
            assert len(prompt) == 1
            prompt = prompt[0]

        assert isinstance(prompt,str),\
            ValueError("Soft Prompt Inference only support single example Inference until now.")
        
       
        messages = [{"role":"system","content":self.system_prompt},
                    {"role":"user","content":prompt}]
        
        input_text = self.promptBuilder(messages)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
        input_embed = self.embedding[input_ids]

        soft_prompt_embd = soft_prompt_embd.type(torch.float32)
        
        if self.linear is None:
            soft_prompt_embd = self.linear(soft_prompt_embd)

        soft_prompt_embd = soft_prompt_embd.reshape(1, self.n_prompt_tokens, -1)
        soft_prompt_embd = soft_prompt_embd.to(device=input_embed.device, dtype=input_embed.dtype)

       
        concat_input_embed = torch.cat((soft_prompt_embd, input_embed), 1)
        
        outputs = self.model.generate(concat_input_embed, self.gen_conf)

        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return LLMInferenceResult(dialog_history=messages,result=result)
             
