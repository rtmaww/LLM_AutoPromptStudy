from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig
#LlamaForCausalLM
import sys

sys.path.append("/root/AutoPrompter/LLM")
from PromptBuilder import LlamaPromptBuilder,VicunaPromptBuilder


def test_Llama():
    
    #model_hf_path = "/apdcephfs_cq2/share_1567347/share_info/llm_models/Llama-2-7b-chat-hf"
    model_hf_path = "/apdcephfs_cq3/share_1567347/xiaoleiwang/model/vicuna-7b-v1.5"
    model = AutoModelForCausalLM.from_pretrained(model_hf_path)
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path,padding_side = "left")
    _dict =  {'n':  1,
            'temperature': 1.0,
            'top_p' :  0.9,
            'max_new_tokens': 400}
        
    gen_conf = GenerationConfig.from_dict(_dict
            )

    prompts = [
        [{"role":"system","content":"Be a helpful assistant."}
        ,{"role":"user","content":"Hello,My name is Xiaolei."}
          ,{"role":"assistant","content":"hi,Xiaolei! Glad to serve for you!"},
          {"role":"user","content":" What is my name?"}],
           [{"role":"system","content":"Be a helpful assistant."}
        ,{"role":"user","content":"Hello,My name is Xiaolei."}
          ,{"role":"assistant","content":"hi,Xiaolei! Glad to serve for you!"},
          {"role":"user","content":" What is my name?"}]
    ]

    
    prompt_builder = VicunaPromptBuilder()
    prompt = [
        prompt_builder.build_prompt(prompts[0]),
        prompt_builder.build_prompt(prompts[1])
        ]
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt",padding=True)
    print(f'input_id_shape:{inputs.input_ids.shape}')
    # Generate gen_conf)
    generate_ids = model.generate(inputs.input_ids,gen_conf)
    print(f'generate_id_shape:{generate_ids.shape}')
    generate_ids = generate_ids[:,inputs.input_ids.shape[1]:]
    print(f'generate_id_shape:{generate_ids.shape}')
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    print(result[0])
    print(result[1])
    
    #result = result[len(prompt):]
    print("\n\n\n")

    #"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."

if __name__ == "__main__":
    test_Llama()