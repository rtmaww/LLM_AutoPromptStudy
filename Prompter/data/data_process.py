    
from data.data_utils import load_data
from config_file import RANDOM_SEED
import random

random.seed(RANDOM_SEED)
def data_prepare(args,prompt_gen_data= True):    
    task, task_cty, HF_cache_dir=args.task, args.task_cty, args.HF_cache_dir
    print(f'Task: {task_cty}_{task}')
    #assert args.task in TASKS, 'Task not found!'

    ini_data,eval_data, test_data = load_data('ini', task, task_cty),load_data('eval', task, task_cty), load_data('test', task, task_cty)

    ini_data = ini_data[0], [output[0] for output in ini_data[1]]
   
    return ini_data, eval_data, test_data

    

    