import sys
sys.path.append("AutoPrompter/LLM")
from llm_inference.LLMInference import LLMInference,LLMInferenceResult



class HFModelInference(LLMInference):

    def __init__(self, promptBuilder=None,gen_kwargs={}) -> None:

        super().__init__(promptBuilder, gen_kwargs)
        assert 'HF_dir' in gen_kwargs,\
            ValueError('loading HFModel need a path which save the weight from huggingface.')
        
        assert promptBuilder is not None,\
            ValueError('HFmodel inference need a promptBuilder.')
        
    
        
        #self.gen_conf = GenerationConfig.from_dict(self.gen_conf)
    
    # def set_gen_conf(self, gen_kwargs):
    #     return super().set_gen_conf(gen_kwargs)

    def generate_text(self, prompts):
        super().generate_text(prompts)
        if not isinstance(prompts,list):
            prompts = [prompts]
            for ind in range(len(prompts)):
                if not isinstance(prompts[ind],list):
                    prompts[ind] = [prompts[ind]]

        batch_size,dialog_turn = prompts.shape
        if self.system_prompt is not None:
            batch_messages = [[{"role":"system","content":self.system_prompt}] for i in range(batch_size)]
        else:
            batch_messages = [[] for i in range(batch_size)]
        trans_messages = [[] for i in range(batch_size)]

        for cur_dialog_turn in range(dialog_turn):
            
            for ind in range(batch_size):
                batch_messages[ind].append({"role":"user","content":prompts[ind,cur_dialog_turn]})
                trans_messages[ind] = self.promptBuilder.build_prompt(batch_messages[ind])
            
            inputs = self.tokenizer(trans_messages, return_tensors="pt",padding=True).to(self.model.device)

            # Generate
            generate_ids = self.model.generate(inputs.input_ids, self.gen_conf)
            generate_ids = generate_ids[:,inputs.input_ids.shape[1]:]
            result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

            for ind in range(batch_size):
                batch_messages[ind].append({"role":"assistant","content":result[ind]})

        return LLMInferenceResult(dialog_history=batch_messages,result= result)
             
