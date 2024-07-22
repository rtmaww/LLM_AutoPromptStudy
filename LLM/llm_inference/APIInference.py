import requests
import copy
import json
import sys
import time
sys.path.append("/AutoPrompter/LLM")
from llm_inference.LLMInference import LLMInference,LLMInferenceResult
from PromptBuilder import LlamaPromptBuilder




class APIInference(LLMInference):
    def __init__(self, promptBuilder=None, gen_kwargs=None) -> None:
        super().__init__(promptBuilder, gen_kwargs)
    
    def generate_text(self,prompt):
        if not isinstance(prompt,list):
            prompt = [prompt]
        return prompt
    
    def api(self,message):
        pass
    
    # def set_gen_conf(self, gen_kwargs):
    #     return super().set_gen_conf(gen_kwargs)



'''
other api LLM to use:
complete the ``generate_text``, ``api`` function
'''

class GPT_Inference(APIInference):
    def __init__(self, promptBuilder=None, gen_kwargs = {}) -> None:

        super().__init__(promptBuilder=promptBuilder,gen_kwargs=gen_kwargs)

        assert gen_kwargs['model'] in ['turbo','gpt-4'],\
            ValueError("Using GPT_Inference the param ``model`` in gen_kwargs  should be set in ['turbo','gpt-4']")
        

        if gen_kwargs['model'] == 'turbo':
            self.api_key = ""
            self.engine = 'gpt-35-turbo-1106'
        elif gen_kwargs['model'] == 'gpt-4':
            self.api_key = ""
            self.engine = 'gpt-4'
        
        self.headers = {
            "X-Api-Key": self.api_key
        }
        self.standard_data = \
        {
            'model': self.engine,
            'messages': [{"role": "system", "content": self.system_prompt}],
            'n': self.gen_conf.n,
            'temperature':self.gen_conf.temperature,
            'top_p' : self.gen_conf.top_p,
            #'max_new_tokens': self.gen_conf.max_new_tokens
        } 
        #self.gen_conf = self.standard_data
    
    # def set_gen_conf(self, gen_kwargs):
    #     return super().set_gen_conf(gen_kwargs)

    def reset(self):
        self.standard_data['messages'] = [
                {"role": "system", "content": self.system_prompt},
                ]
        

    def api(self,message):
        response = None
        filter_n = 0
        limit_error_times = 0
        while response is None and filter_n < 10:
            try:
                #print(message)
                #print(requests.post("http://9.135.143.158:8998/chat1", json=message, headers=self.headers))
                
                response = requests.post("http://9.134.231.57:8998/chat1", json=message, headers=self.headers).json()
                
                #response = json.loads(response)
                assert 'choices' in response
                for i in range(len(response['choices'])):
                    assert(response['choices'][i]['finish_reason']=="stop")
                    assert('content' in response['choices'][i]['message'])
                return response
            except Exception as e:
                try:
                    if response["error"]["code"] == "too many requests":
                        print(response)
                        time.sleep(20)
                        limit_error_times += 1
                except:
                    pass
                    
                    
                filter_n += 1
                print(e)
                print(message)
                print(response)
                response = None
        return {"choices":[{"message":{"content":""}}]}

    def generate_text(self,prompt):

        prompt = super().generate_text(prompt)
        #self.reset()
        message = copy.deepcopy(self.standard_data)


        for sub_prompt in prompt:
            message['messages'].append({"role": "user", "content": sub_prompt})
            #print(message)
            result = self.api(message=message)
            message['messages'].append({"role": "assistant", "content": result["choices"][0]["message"]["content"]})

        answer = [item["message"]["content"] for item in result["choices"]]     
        history = message['messages']
        #self.reset()
        return LLMInferenceResult(dialog_history=history,result=answer)
    

class Llama_Inference(APIInference):

    def __init__(self, promptBuilder=None, gen_kwargs={}) -> None:
        super().__init__(promptBuilder, gen_kwargs)
        self.promptBuilder = LlamaPromptBuilder()
        
        assert gen_kwargs['model'] in ['Llama_7B','Llama_13B','Llama_70B'],\
            ValueError("gpt model should be set in ['Llama_7B','Llama_13B','Llama_70B']")
        

        if gen_kwargs['model'] == 'Llama_7B':
            self.api_ = 'http://11.198.27.49:8081/generate'
        elif gen_kwargs['model'] == 'Llama_70B':
            self.api_ = "http://9.91.15.176:8081/generate"
        else:
            pass
        
       
        self.standard_data = \
        {
            'prompt': [{"role": "system", "content": self.system_prompt}],
            "top_p": 1.0,
            "temperature": 0,
            "top_k": -1,
            "max_tokens": 1024,

        } 
    
    def reset(self):
        self.standard_data['prompt'] = [
                {"role": "system", "content": self.system_prompt},
                ]
        

    def api(self,message):
        response = None
        query_msg = copy.deepcopy(message)
        query_msg['prompt'] = self.promptBuilder.build_prompt(message['prompt'])
        while response is None:
          
            response = requests.post(self.api_, json = query_msg).json()
            
            #print(f'response:{response}')
        return response['text'][0]

    def generate_text(self,prompt):

        prompt = super().generate_text(prompt)
        #self.reset()
        message = copy.deepcopy(self.standard_data)

        for sub_prompt in prompt:
            message['prompt'].append({"role": "user", "content": sub_prompt})
            result = self.api(message=message)
            message['prompt'].append({"role": "assitant", "content": result})

        answer = [result]     
        history = message['prompt']
        return LLMInferenceResult(dialog_history=history, result=answer)

class Claude_Inference(APIInference):
    '''
    ``todo``
    '''
    
    def __init__(self, promptBuilder=None, gen_kwargs={}) -> None:
        super().__init__(promptBuilder, gen_kwargs)

    def generate_text(prompts, cur_thread=0, share_buffer=None):
        return super().generate_text(cur_thread, share_buffer)
    
   

    def api(self):
        return super().api()
