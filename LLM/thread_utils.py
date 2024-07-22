import threading
import numpy as np
import functools
import sys
sys.path.append("/Users/wangxiaolei/Downloads/AutoPrompter/LLM")
from llm_inference.LLMInference import LLMInference
from llm_inference.APIInference import APIInference
from LLM.llm_inference.APIInference import GPT_Inference,Llama_Inference
import concurrent
import time
from config_file import THREAD_MAX_WORKERS


pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_MAX_WORKERS)
thread_Lock = threading.Lock()

def timeout(seconds=60, error_message="Timeout"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result_container = [None]
            exception_container = [None]
            thread_stop = threading.Event()

            def target():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
                thread_stop.set()

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                thread_stop.set()
                raise TimeoutError(error_message)
            if exception_container[0] is not None:
                raise exception_container[0]
            return result_container[0]

        return wrapper

    return decorator

def mutiThreadInference(prompts:list,inferencer:LLMInference):#,gen_kwargs:dict):
    


    #inferencer.set_gen_conf(gen_kwargs=gen_kwargs)
    if isinstance(inferencer,Llama_Inference):
        THREAD_MAX_WORKERS = 20
    elif isinstance(inferencer,GPT_Inference):
        THREAD_MAX_WORKERS = 1
        
    if not isinstance(prompts,list):
        prompts = [prompts]
    # for prompt in prompts:
    #     print(prompt)
    prompt_len = len(prompts)
    batch_size = max(1,int(prompt_len/THREAD_MAX_WORKERS))
    batch_prompts = [prompts[i:i+batch_size] for i in range(0,prompt_len,batch_size)]
    thread_num = len(batch_prompts)
    print(f"There has total {prompt_len} queries, SYS devide in {thread_num} thread to Query LLM. each Thread has {batch_size} queries!")
    
    share_buffer = [[] for i in range(thread_num)]
    futures = []

    @timeout(seconds=300)
    def call_function(prompt,inferencer):
        return inferencer.generate_text(prompt)


    def call_function_batch(batch_prompts,inferencer,thread_ind,share_buffer):
        results =[]
        for single_prompt in batch_prompts:
            flag = True
            while flag:
                try:
                    infer_result = call_function(single_prompt,inferencer)
                    flag = False
                except Exception as e:
                    print(e)
            results.append(infer_result)

        acquired = thread_Lock.acquire(blocking=False)
        while not acquired:
            acquired = thread_Lock.acquire(blocking=False)
        share_buffer[thread_ind] = results
        thread_Lock.release()
        return 

    
    

    start_time = time.time()

    for thread_ind in range(thread_num):
       
        future = pool.submit(
            call_function_batch,
            batch_prompts[thread_ind], 
            inferencer,
            thread_ind,
            share_buffer)
        
        futures.append(future)
    
    for future in concurrent.futures.as_completed(futures):
        pass

  
    end_time = time.time()
    print(f"api time consuming:{end_time-start_time}")
    agg_buffer = []
    for local_buffer in share_buffer:
        agg_buffer.extend(local_buffer)
    #share_buffer= np.array(share_buffer).reshape(-1).tolist()
    share_buffer = agg_buffer
    share_buffer_history = [item.history for item in share_buffer ]
    share_buffer_result = [item.result for item in share_buffer]

    
    share_buffer_result = np.array(share_buffer_result).reshape(-1).tolist()
    #share_buffer_history = np.array(share_buffer_history).reshape(-1).tolist()

    return share_buffer_history,share_buffer_result
