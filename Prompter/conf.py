Initial_conf = {
    ## initializer kwargs
    "initial_prompt":"",
    "prompt_gen_data":"",
    "fewshot":5,
    "seed_file_path":"",
    ##gen_config
}


API_inference_conf = {
    "model":"",
    "n":1,

}
HF_inference_conf = {
    "HF_dir":""
    }
soft_prompt_inference_conf = {
    "n_prompt_tokens":5,
    "intrinsic_dim":10,
    "random_proj":"uniform",
}

evaluation_gen_conf = {
    "temperature": 0,
    "top_p":1.0
}

generation_gen_conf = {
    "temperature": 1.0,
    "top_p":0.9
}

def merge_conf(conf_1=None,conf_2=None):
    return_conf = conf_1.copy()
    for k,v in conf_2.items():
        return_conf[k] = v
    return return_conf