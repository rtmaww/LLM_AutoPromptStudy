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



We divide Autoprompter into the following components: Initializer, Evaluator, Optimizer, Scorer, Selector, Template. All required components have been defined in the base class. You only need to modify the parts that need to be customized. For Evaluator and Optimizer, you You only need to modify the corresponding inferencer class and pass it into the reconstruction
