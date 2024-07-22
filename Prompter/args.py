import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Autoprompt pipeline")
    
    
    parser.add_argument(
        "--task_cty",
        type=str,
        default="bigbench-ii",
        help="The parent folder of task data",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="navigate",
        help="The name of the dataset to use (via the datasets library).",
    )
   
    parser.add_argument(
        "--initial_stylization",
        type=bool,
        default=False,
        help="give some prompt style to refer in prompt initialization."
    )
    
    
    
    
    
   
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default=None,
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."    
    )
   
    parser.add_argument(
        "--eval_model",
        type=str,
        default='turbo',
        help="the api model for evaluation"    
    )
    
    parser.add_argument(
        "--ops_method",
        type=str,
        default='apo',
        help="your customized LLM prompt optimization method. now support methods in [apo,ape,apo_sum,apo_agent,ORPO]"    
    )
    
        
    
    parser.add_argument(
        "--STORE_BATCH_SIZE",
        type=int,
        default=5,
        help="The number of prompts retained at each step ",
    )
    
    parser.add_argument(
        "--GEN_BATCH_SIZE",
        type=int,
        default=2,
        help="The number of “example sampling taken” during initialization",
    )
    
    parser.add_argument(
        "--GEN_SUBSAMPLE",
        type=int,
        default=4,
        help="The number of examples selected in per sampling during initialization",
    )
    
    parser.add_argument(
        "--GEN_DATA_NUMBER",
        type=int,
        default=2,
        help="the number of answers returned by each query when sending an api request.",
    )
    
    parser.add_argument(
        "--OPS_STEP",
        type=int,
        default=10,
        help="the number of steps for optimization.",
    )
     
    parser.add_argument(
        "--EAVL_DATA_NUMBER",
        type=int,
        default=50,
        help="the number of examples for evaluation.",
    )
      
    parser.add_argument(
        "--TEST_DATA_NUMBER",
        type=int,
        default=200,
        help="the number of examples for test.",
    )
    
    args = parser.parse_args()
    return args

def check_params(args):
    pass