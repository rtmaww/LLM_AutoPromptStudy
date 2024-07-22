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




# Customization Guide（quickly implementation your custom AutoPrompter）

## Step 1: Customize the Initialization/Evaluation/Optimization Template

To customize the initialization, evaluation, or optimization template, you need to imitate the implementation of the base class in `Prompter/template`. Define slots in the template and complete the `fill()` function as follows:

```python
class PromptScoreTemplate:
    def __init__(self) -> None:
        self.template = "text: [INS]\nscore: [SCORE]\n"
        self.delimiter = "\n"
        self.slots = ["[INS]", "[SCORE]"]

    def fill(self, score_dict):
        template = ""
        length = len(score_dict)
        cur_count = 1
        for prompt, score in score_dict.items():
            template += self.template.replace('[INS]', prompt).replace('[SCORE]', str(int(score * 100)))
            if cur_count != length:
                template += self.delimiter
            cur_count += 1
        return template
```
## Step 2: Extend Autoprompter

We divide Autoprompter into the following components: Initializer, Evaluator, Optimizer, Scorer, Selector, and Template. All required components have been defined in the base class. You only need to modify the parts that need customization. For Evaluator and Optimizer, you only need to modify the corresponding inferencer class and pass it into the reconstruction.

Specifically, extend the Autoprompter class and rewrite the optimize() function. Pass the contents of the template slots defined in Step 1 into kwargs for calling.

```python
class LLM_AS_Optimizer_Prompter(AutoPrompter):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gen_inferencer = GPT_Inference(
            promptBuilder=None,
            gen_kwargs=self.ops_infer_conf
        )
        
        # [EXAMPLE, INS_SCORE]
        self.optimization_template = template.Ops_Prompt_Score_Example_Template()

    def optimize(self):
        # Your own logic
        optimize_kwargs = {
            "[INS_SCORE]": ins_score_dict,
            "[EXAMPLE]": error_example[ind][:2]
        }

        history, instructions = self.optimizer.prompt_optimize(
            template=self.optimization_template,
            prompts=None,
            optimize_kwargs=optimize_kwargs
        )
```

## Step 3: (Optional) Implement Custom Initializer, Scorer, and Selector

Inherit the base class to implement your own Initializer, Scorer, and Selector when your modifications do not only involve the template. For example, if the scorer changes from exact match to likelihood, or the selector changes from top-n to Monte Carlo, etc.

For more details, you can refer to Prompter/AutoPrompter.py.


# How to Use
set your data for initialization/evaluation/optimization in "/AutoPrompter/LLM_AutoPromptStudy/Prompter/data/"

```shell
python3 Prompter/AutoPrompter --task_cty "bigbench-ii" \
                    --task "navigate" \
                    --initial_stylization False \
                    --eval_model turbo \
                    --ops_method "APE" \
                    --STORE_BATCH_SIZE 5 \
                    --GEN_BATCH_SIZE 5 \
                    --GEN_SUBSAMPLE 2 \
                    --GEN_DATA_NUMBER 5 \
                    --EAVL_DATA_NUMBER 50 \
                    --TEST_DATA_NUMBER 200 \

```
## Our params in experiment

| ops_method | STORE_BATCH_SIZE | GEN_BATCH_SIZE | GEN_SUBSAMPLE | GEN_DATA_NUMBER | OPS_step |
| :----:| :----: | :----: |  :----: | :----: | :----: |
| APE | 5 | 5 | 2 | 5 | 10 |
| APO | 5 | 5 | 2 | 1 | 10 |
| APO_Agent | 5 | 5 | 2 | 1 | 10 |
| APO_Sum | 5 | 5 | 2 | 2 | 10 |
| ORPO | 5 | 5 | 2 | 5 | 10 |



##  manage the middle result of AutoPrompter
```shell
python3 Prompter/instruction_tree.py
```
![RUNOOB 图标](https://static.jyshare.com/images/runoob-logo.png)

