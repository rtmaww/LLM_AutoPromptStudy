# Code Framework

## LLM

- **llm_inference**: Supports multiple LLM inferences. You can call it through the API or load models from Huggingface for inference.
- **PromptBuilder**: A base class for prompt wrapping. You can inherit this base class to wrap prompts for different models. Currently, this code is implemented for Llama and Vicuna.
- **thread_utils**: Implements multi-threaded API calls for LLM.

## Prompter

- **data**: Handles data processing for tasks in BigBench.
- **initializer**: A base class for prompt initialization.
- **evaluation**: A base class for Evaluator and Scorer. You can inherit both to customize your own methods for evaluation and scoring.
- **optimization**: A base class for LLM optimization.
- **template**: Contains all templates for the LLM prompter.
- **Selector**: You can inherit the selector base class to customize your own filter strategy, such as Monte Carlo.
- **AutoPrompter**: A base Prompter class that includes Initializer, Evaluator, Scorer, Selector, Optimizer, and Template. You can extend this class and override the `optimizer` function to customize your own optimization method. Currently implemented methods include:
  - `APO_Prompter`
  - `APE_Prompter`
  - `LLM_AS_Optimizer_Prompter`
  - `APO_Agent_Prompter`
  - `Random_Hint`
- **instruction_tree**: Provides a visual management tool for autoprompt, supporting the visualization of error samples, feedback, and other intermediate results during the prompt optimization process.


# How to quickly implement your custom AutoPrompter

step 1. customize the Initialization/evaluation/optimization template, you need to imitate the implementation of the base class in Prompter/template, define slots in the template, and complete the fill() function as follows:
```python
class PromptScoreTemplate():
    def __init__(self) -> None:
        self.template = "text: [INS]\nscore: [SCORE]\n"
        self.delimiter = "\n"
        self.slots = ["[INS]","[SCORE]"]
    def fill(self,score_dict):
        template = ""
        
        length = len(list(score_dict.keys()))
        cur_count = 1
        for prompt,score in score_dict.items():
            template += self.template.replace('[INS]',prompt).replace('[SCORE]',str(int(score*100)))
            if cur_count != length:
                template += self.delimiter
        return template
```

Step 2 . We divide Autoprompter into the following components: Initializer, Evaluator, Optimizer, Scorer, Selector, Template. All required components have been defined in the base class. You only need to modify the parts that need to be customized. For Evaluator and Optimizer, you You only need to modify the corresponding inferencer class and pass it into the reconstruction.
Specifically, Extend Autoprompter Class, Rewrite the optimize() function. Pass the contents of the template slots defined in step 1 into kwargs for calling.

```python
class LLM_AS_OPtimizer_prompter(AutoPrompter):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gen_inferencer = GPT_Inference(promptBuilder=None,
                                            gen_kwargs=self.ops_infer_conf)
        
        ##[EXAMPLE,INS_SCORE]
        self.optimization_template = template.Ops_Prompt_Score_Example_Template()
    def optimize():
          "your own logic"
          optimize_kwargs = {
                "[INS_SCORE]":ins_score_dict,
                "[EXAMPLE]":error_example[ind][:2]
            }

          history,instructions = self.optimizer.prompt_optimize(
                template=self.optimization_template,
                prompts=None,
                optimize_kwargs=optimize_kwargs
                )
  
```
Step 3 .(Optional)




