import json



class Template(object):
    def __init__(self) -> None:
        self.template = ""
        self.slots = []
    def fill(self,slots,template=None):
        if template is None:
            template = self.template
        for slot_name in self.slots:
            
            template = template.replace(slot_name,slots[slot_name])
        return template


class ExampleTemplate():

    def __init__(self) -> None:
        self.delimiter = "\n\n"
        self.template = "## Input\n[INPUT]\n## Output\n[OUTPUT]"
        self.list_template = "### Sample [INDEX]\n## Input\n[INPUT]\n## Output\n[OUTPUT]"

    def fill(self,examples,test=False,wrap = True):
        template = ""
        inputs,outputs = examples[0],examples[1]
        if not isinstance(inputs,list):
            inputs = [inputs]
            outputs = [outputs]
        
        if test:
            outputs = ["" for i in range(len(inputs))]
        elif wrap:
            
            new_outputs = []
            for output in outputs:
                store = {"answer":output[0]} if isinstance(output,list) else {"answer":output}
                new_outputs.append(json.dumps(store))
            outputs = new_outputs

        if len(inputs) > 1:
          
            for sample_ind,(input,output) in enumerate(zip(inputs,outputs)):
                if isinstance(output,list):
                    template += self.list_template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[OUTPUT]',output[0])
                else:
                    template += self.list_template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[OUTPUT]',output)
                if sample_ind != len(inputs)-1:
                    template += self.delimiter
        else:
            if isinstance(outputs[0],list):
                template += self.template.replace('[INPUT]',inputs[0]).replace('[OUTPUT]',outputs[0][0])
            else:

                template += self.template.replace('[INPUT]',inputs[0]).replace('[OUTPUT]',outputs[0])

        return template

class Example_Ins_Pos_Template():

    def __init__(self) -> None:
        self.delimiter = "\n\n"
        self.Q_begin_template = "## Input\n<INS>[INPUT]\n## Output\n[OUTPUT]"
        self.Q_begin_list_template = "### Sample [INDEX]\n## Input\n<INS>[INPUT]\n## Output\n[OUTPUT]"

        self.Q_end_template = "## Input\n[INPUT]<INS>\n## Output\n[OUTPUT]"
        self.Q_end_list_template = "### Sample [INDEX]\n## Input\n[INPUT]<INS>\n## Output\n[OUTPUT]"

        self.A_begin_template = "## Input\n[INPUT]\n## Output\n<INS>[OUTPUT]"
        self.A_begin_list_template = "### Sample [INDEX]\n## Input\n[INPUT]\n## Output\n<INS>[OUTPUT]"

    def fill(self,examples,test=False,wrap = True,ins_pos = "Q_begin"):
        if ins_pos == "Q_begin":
            self.template = self.Q_begin_template
            self.list_template = self.Q_begin_list_template
        elif ins_pos == "Q_end":
            self.template = self.Q_end_template
            self.list_template = self.Q_end_list_template
        elif ins_pos == "A_begin":
            self.template = self.A_begin_template
            self.list_template = self.A_begin_list_template

        template = ""
        inputs,outputs = examples[0],examples[1]
        if not isinstance(inputs,list):
            inputs = [inputs]
            outputs = [outputs]
        
        if test:
            outputs = ["" for i in range(len(inputs))]
        elif wrap:
            
             outputs = ['{\"answer\":'+f'\"{output[0]}\"'+'}' if isinstance(output,list) else '{\"answer\":'+f'\"{output}\"'+'}' \
                        for output in outputs]

        if len(inputs) > 1:
            
            for sample_ind,(input,output) in enumerate(zip(inputs,outputs)):
                if isinstance(output,list):
                    template += self.list_template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[OUTPUT]',output[0])
                else:
                    template += self.list_template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[OUTPUT]',output)
                if sample_ind != len(inputs)-1:
                    template += self.delimiter
        else:
            if isinstance(outputs[0],list):
                template += self.template.replace('[INPUT]',inputs[0]).replace('[OUTPUT]',outputs[0][0])
            else:

                template += self.template.replace('[INPUT]',inputs[0]).replace('[OUTPUT]',outputs[0])

        return template

class ErrorExampleTemplate():
    def __init__(self) -> None:
        self.template = "### Sample [INDEX]\n## Input\n[INPUT]\n## Correct Answer\n[Answer]\n## Output\n[OUTPUT]"
        self.delimiter = "\n\n"
    def fill(self,examples):
        if examples == []:
            return "None"
        
        template = ""
        inputs,outputs,llm_outputs = examples[0],examples[1],examples[2]
        if not isinstance(inputs,list):
            inputs = [inputs]
            outputs = [outputs]
        
        
        if len(inputs) > 1:
            
            for sample_ind,(input,output,llm_output) in enumerate(zip(inputs,outputs,llm_outputs)):
                if isinstance(output,list):
                    template += self.template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[Answer]','{\"answer\":\"'+output[0]+'\"}').replace('[OUTPUT]',llm_output)
                else:
                    template += self.template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[Answer]','{\"answer\":\"'+output+'\"}').replace('[OUTPUT]',llm_output)
                if sample_ind != len(inputs)-1:
                    template += self.delimiter
        else:
            if isinstance(outputs[0],list):
                template += self.template.replace('[INPUT]',inputs[0]).replace('[Answer]','{\"answer\":\"'+outputs[0][0]+'\"}').replace('[OUTPUT]',llm_outputs[0])
            else:
                template += self.template.replace('[INPUT]',inputs[0]).replace('[Answer]','{\"answer\":\"'+outputs[0]+'\"}').replace('[OUTPUT]',llm_outputs[0])

        return template

class PromptScoreTemplate():
    def __init__(self) -> None:
        self.template = "text: [INS]\nscore: [SCORE]\n"
        self.delimiter = "\n\n"
    def fill(self,score_dict):
        template = ""
        
        length = len(list(score_dict.keys()))
        cur_count = 1
        for prompt,score in score_dict.items():
            template += self.template.replace('[INS]',prompt).replace('[SCORE]',str(int(score*100)))
            if cur_count != length:
                template += self.delimiter
        return template

class PromptSeedTemplate():
    def __init__(self) -> None:
        self.template = \
'''Your instruction should include asking the assistant to [CONSTRAINT].

Instruction example:
[INS_EXAMPLE]
'''
    def fill(self,seed_ins_dict):
        templates = []
        for constraint,ins_example in seed_ins_dict.items():
             templates.append(self.template.replace('[CONSTRAINT]',constraint).replace('[INS_EXAMPLE]',ins_example))
        return templates
             


class Initial_Prompt_Fewshot_Template(Template):
    def __init__(self) -> None:
        self.template = \
'''Following is several examples of a task. The input of each sample are following \"## Input\",and the outputs are following \"## Output\".

[EXAMPLE]

Please read the description and samples and write a BRIEF instruction of this task. The instruction should also specify the formatting of the answer.
Instruction:'''
        print(self.template)
        self.slots = ['[EXAMPLE]']
    def fill(self,slots,template= None):
        example_template = ExampleTemplate()
        slots['[EXAMPLE]'] = example_template.fill(examples=slots['[EXAMPLE]'])
        return super().fill(slots,self.template)
    
class Initial_Prompt_Desc_Fewshot_Template(Initial_Prompt_Fewshot_Template):
    def __init__(self) -> None:
        self.template = \
'''Following is a task description and several samples of a task. The input of each sample are following "## Input",and the outputs are following "## Output".

### Task Description
[DESC]

### Samples
[EXAMPLE]


Please read the description and samples and write a BRIEF instruction of this task. The instruction should also specify the formatting of the answer.

Instruction:
'''
        self.slots = ['[DESC]','[EXAMPLE]']
    def fill(self,slots,template = None):
        return super().fill(slots,self.template)
    
class Initial_Prompt_Desc_Fewshot_COT_Template(Initial_Prompt_Desc_Fewshot_Template):
    def __init__(self) -> None:
        super().__init__()
        self.template = \
'''Following is a task description and several samples of a task. The input of each sample are following "## Input",and the outputs are following "## Output".

### Task Description
[DESC]

### Samples
[EXAMPLE]


You should:
1. generate a concise multi-step instruction to complete the task. 
2. The instruction must explicitly ask the follower to write down the thinking process.
3. Remember to specify the answer format.

Instruction:
'''
    def fill(self, slots, template=None):
        return super().fill(slots, template)

class Initial_Prompt_Fewshot_Seed_Template(Template):
    def __init__(self) -> None:
         self.template = \
'''Following is several examples of a task. The input of each sample are following \"## Input\",and the outputs are following \"## Output\".

[EXAMPLE]

Please read the description and samples and write a BRIEF instruction of this task. [SEED_DICT]

Instruction:'''
# Please read the description and samples and write a BRIEF instruction of this task. The instruction should also specify the formatting of the answer. 
# [SEED_DICT]
    def fill(self, slots, template=None):
        templates = []
        example_template = ExampleTemplate()
        slots['[EXAMPLE]'] = example_template.fill(examples=slots['[EXAMPLE]'])
        seed_strings = slots['[SEED_DICT]']
        for seed_string in seed_strings:
            slots['[SEED_DICT]'] = seed_string
            templates.append(super().fill(slots,self.template))
             


        return templates

class Initial_Multi_Prompt_Desc_Fewshot_Template(Initial_Prompt_Fewshot_Template):
    def __init__(self) -> None:
        self.template = \
'''Following is a task description and several samples of a task. The input of each sample are following "## Input",and the outputs are following "## Output".

### Task Description
[DESC]

### Samples
[EXAMPLE]

You are a teacher. You should write a prompt for this task for your student. The prompt should consist of:
<START>
Task description
Keypoints for avoiding possible mistakes
Helpful tips
Answer Format
<END>
'''
        self.slots = ['[DESC]','[EXAMPLE]']
    def fill(self,slots,template = None):
        return super().fill(slots,self.template)


    
class Initial_Prompt_Desc_Fewshot_Seed_Template(Initial_Prompt_Fewshot_Seed_Template):
    def __init__(self) -> None:
        self.template = \
'''Following is a task description and several samples of a task. The input of each sample are following \"## Input\",and the outputs are following \"## Output\".

### Task Description
[DESC]

### Samples
[EXAMPLE]

[SEED_DICT]

Instruction:
'''
        self.slots = ['[DESC]','[EXAMPLE]','[SEED_DICT]']
    def fill(self, slots, template=None):
         return super().fill(slots, template)

class initial_InstructZero_template(Initial_Prompt_Desc_Fewshot_Template):
    def __init__(self) -> None:
        super().__init__()
        self.template = \
'''Following is a task description and several samples of a task. The input of each sample are following "## Input",and the outputs are following "## Output".

### Task Description
[DESC]

### Samples
[EXAMPLE]

The Instruction is:
'''      
        self.slots = ['[EXAMPLE]','[DESC]']
    def fill(self, slots, template=None):
        return super().fill(slots, template)

class EvaluationTemplate(Template):
    def __init__(self) -> None:
        self.template = \
"[INS]\n\n[EXAMPLE]"

        self.slots = ['[INS]','[EXAMPLE]']
    def fill(self,slots,template=None):
        #examples = slots['[]']
        example_template = ExampleTemplate()
        slots["[EXAMPLE]"] = example_template.fill(examples=slots["[EXAMPLE]"],test=True)

        return super().fill(slots,self.template)


    
class Ops_Paraphrase_Plus_Template(Template):
    def __init__(self) -> None:
        self.template = \
'''Here are several examples for a task:

[EXAMPLE]

Now I have an instruction for this task:
Instruction:

[INS]

Please generate a variation of this instruction while keeping the semantic meaning.

New instruction:'''
        self.slots = ['[INS]','[EXAMPLE]']
    def fill(self,slots,template=None):
        example_template = ExampleTemplate()
        slots['[EXAMPLE]'] = example_template.fill(examples=slots['[EXAMPLE]'])
        return super().fill(slots)



class Zero_COT_EvaluationTemplate(Template):
    def __init__(self) -> None:
        self.template = \
"[INS]\n\n[EXAMPLE]\n\n## Output:\nLet's think step by step."



        self.slots = ['[INS]','[EXAMPLE]']
    def fill(self,slots,template=None):
        #examples = slots['[]']
        example_template = ExampleTemplate()
        slots["[EXAMPLE]"] = example_template.fill(examples=slots["[EXAMPLE]"],test=True)

        return super().fill(slots,self.template)





class Ops_ParaphraseTemplate(Template):
    def __init__(self) -> None:
        self.template = \
'''Generate a variation of the following instruction while keeping the semantic meaning.
Input: 
[INS]

Output:'''

        self.slots = ['[INS]']
    def fill(self,slots,template=None):
        return super().fill(slots)
        
class Ops_Query_Feedback_Template(Template):
    def __init__(self) -> None:
        self.template = \
'''
My current prompt is: 
\"[INS]\" 

But it gets the following examples wrong: 
[ERROR_EXAMPLE]

give 3 reasons why the prompt could have gotten these examples wrong.
Wrap each reason with <START> and <END>.
'''
        self.slots = ['[INS]','[ERROR_EXAMPLE]']

    def fill(self,slots,template=None):
        error_example_template = ErrorExampleTemplate()
        slots['[ERROR_EXAMPLE]'] = error_example_template.fill(slots['[ERROR_EXAMPLE]'])
        return super().fill(slots,self.template)

class Ops_Update_Feedback_Template(Template):
    def __init__(self) -> None:
        self.template = \
'''
My current prompt is: 
\"[INS]\"  

But it gets the following examples wrong:
[EXAMPLE]

Based on these examples the problem with this prompt is that :
[FEEDBACK]

Based on the above information, I wrote 2 different improved prompts. Each prompt is wrapped with <START> and <END>. 

The 2 new prompts are:
'''
        self.slots = ['[INS]','[EXAMPLE]','[FEEDBACK]']
    def fill(self,slots,template=None):
        example_template = ExampleTemplate()
        slots["[EXAMPLE]"] = example_template.fill(examples=slots["[EXAMPLE]"])
        return super().fill(slots,self.template)



class Error_Example_Template_Agent(Template):
    def __init__(self) -> None:
        self.template = '''[INDEX]
The model's input is:
[INPUT]
The model's response is:
[OUTPUT]
The correct label is: [Answer]
The model's prediction is [E_OUTPUT]'''
        self.delimiter = "\n\n"
    
    def extract_format_output(self, input_text):
        import re
        
        pattern = r'({.*?})'
        pattern2 = r':(.*?)}'
        matches = re.findall(pattern, input_text, re.DOTALL)

        format_error = False
        result = ""
        if matches:
            for match in matches:
                try:
                    match = json.loads(match)
                    for key in ["answer"]:
                        if key in match:
                            result = str(match[key])
                except:
                    matches2 = re.findall(pattern2, match, re.DOTALL)
                    if matches2:
                        result = matches2[-1].strip()

        if not result:

            result = input_text
            if result.endswith("."):
                result = result[:-1]
            format_error = True

        return result

    def fill(self,examples):
        if examples == []:
            return "None"
        
        template = ""
        inputs,outputs,llm_outputs = examples[0],examples[1],examples[2]
        if not isinstance(inputs,list):
            inputs = [inputs]
            outputs = [outputs]
        
        
        if len(inputs) > 1:
            
            for sample_ind,(input,output,llm_output) in enumerate(zip(inputs,outputs,llm_outputs)):
                if isinstance(output,list):
                    template += self.template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[Answer]',output[0]).replace('[OUTPUT]',llm_output).replace('[E_OUTPUT]',self.extract_format_output(llm_output))
                else:
                    template += self.template.replace('[INDEX]',str(sample_ind+1)).replace('[INPUT]',input).replace('[Answer]',output).replace('[OUTPUT]',llm_output).replace('[E_OUTPUT]',self.extract_format_output(llm_output))
                if sample_ind != len(inputs)-1:
                    template += self.delimiter
        else:
            if isinstance(outputs[0],list):
                template += self.template.replace('[INPUT]',inputs[0]).replace('[Answer]',outputs[0][0]).replace('[OUTPUT]',llm_outputs[0]).replace('[E_OUTPUT]',self.extract_format_output(llm_output))
            else:
                template += self.template.replace('[INPUT]',inputs[0]).replace('[Answer]',outputs[0]).replace('[OUTPUT]',llm_outputs[0]).replace('[E_OUTPUT]',self.extract_format_output(llm_output))

        return template
    
class Ops_Query_Feedback_For_Agent(Template):
    def __init__(self) -> None:
        self.template = \
'''I'm writing prompts for a language model designed for a task.

My current prompt is: 
[INS]
But this prompt gets the following examples wrong:
[ERROR_EXAMPLE]
For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.'''

        self.slots = ["[INS]","[ERROR_EXAMPLE]"]
    
    def fill(self,slots):
        error_example_template = Error_Example_Template_Agent()
        slots['[ERROR_EXAMPLE]'] = error_example_template.fill(slots['[ERROR_EXAMPLE]'])
        return super().fill(slots,self.template)
        

class Ops_Update_For_Agent(Template):
    def __init__(self) -> None:
        self.template = \
'''I'm writing prompts for a language model designed for a task.
My current prompt is:
[INS]
But this prompt gets the following examples wrong:
[ERROR_EXAMPLE]
Based on these errors, the problems with this prompt and the reasons are:
[FEEDBACK]
There is a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
[Former_Prompts]
Based on the above information, please write 2 new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START>and <END>. 

The new prompts are:
'''

        self.slots = ['[INS]','[ERROR_EXAMPLE]','[FEEDBACK]','[Former_Prompts]']
    def fill(self,slots,template=None):
        example_template = Error_Example_Template_Agent()
        slots["[ERROR_EXAMPLE]"] = example_template.fill(examples=slots["[ERROR_EXAMPLE]"])
        return super().fill(slots,self.template)


class Ops_Prompt_Score_Example_Template(Template):
    def __init__(self) -> None:
        self.template = \
'''Your task is to generate the instruction <INS>. Below are some previous instructions with their scores. The score ranges from 0 to 100. 

[INS_SCORE]

Below are some problems.

Problems:
[EXAMPLE]

Generate an instruction that is different from all the instructions <INS> above, and has a higher score than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>. The instruction should be concise, effective, and generally applicable to all problems above.
'''
        self.slots = ['[INS_SCORE]','[EXAMPLE]']
    def fill(self, slots,template=None):
        ins_score_template = PromptScoreTemplate()
        slots['[INS_SCORE]'] = ins_score_template.fill(slots['[INS_SCORE]'])
        #example_template = ExampleTemplate()
        example_template = Example_Ins_Pos_Template()
        slots["[EXAMPLE]"] = example_template.fill(examples=slots["[EXAMPLE]"])
                

        return super().fill(slots,self.template)

class Ops_Error_Category_Template(Ops_Query_Feedback_Template):
    def __init__(self) -> None:
        super().__init__()
        self.template = \
'''I have an instruction for a task:
"
[INS]
"

However, this instruction got the following samples wrong ("## Input" is followed by the input, "## Output" is followed by the follower output. "## Correct Answer" is followed by the correct answer.):

[ERROR_EXAMPLE]


Now, I need you to:
1. Read each error sample and carefully analyze why the follower makes mistake. 
2. Group the causes of errors into several categories. Based on the categories, provide several suggestions for improving the instruction.
3. Based on the analysis, please SLIGHTLY revise the instruction to a better one. Wrap the new instruction with <START> and <END>.'''
    def fill(self, slots, template=None):
        return super().fill(slots, template)



class Ops_Error_Behavior_Template(Template):
    def __init__(self) -> None:
        super().__init__()
        self.template = \
'''I have an instruction for a task:
"
[INS]
"
However, this instruction got the following samples wrong ("## Input" is followed by the input, "## Output" is followed by the follower output. "## Correct Answer" is followed by the correct answer.):

[ERROR_EXAMPLE]

Please read the thinking process of the follower and analyze:
1. Analyze at which step did the follower make mistakes in each example.
2. Provide 3 suggestions for further breaking down the solution at the failed steps. 
3. Based on the suggestions, write a new step-by-step instruction that can effectively guide the followers towards successfully completing the task. Wrap the new instruction with <START> and <END>.
'''

        self.slots = ['[INS]','[ERROR_EXAMPLE]']

    def fill(self,slots,template=None):
        error_example_template = ErrorExampleTemplate()
        slots['[ERROR_EXAMPLE]'] = error_example_template.fill(slots['[ERROR_EXAMPLE]'])
        return super().fill(slots,self.template)


class Generate_Ins_behavior_Template(Template):
    def __init__(self) -> None:
        super().__init__()
        self.template = \
'''STRICTLY follow every detail of the following instruction and write out all the thinking process BEFORE writing the answer. 

### Instruction
[INS]

[EXAMPLE]
''' 
        self.slots = ['[INS]','[EXAMPLE]']

    def fill(self, slots):
        example_template = ExampleTemplate()
        slots['[EXAMPLE]'] = example_template.fill(examples=slots['[EXAMPLE]'],test=True)
        return super().fill(slots)


class Concat_Ins_behavior_Template_Turbo(Template):
    def __init__(self) -> None:
        super().__init__()
        self.template = \
'''[INS]

#### Examples
Here are some examples to help you thinking. DONOT simply copy the outputs in the examples!

[EXAMPLE]
[ANSWER]
'''
        self.slots = ['[INS]','[EXAMPLE]','[ANSWER]']
    def fill(self, slots):
        example_template = ExampleTemplate()
        slots['[EXAMPLE]'] = example_template.fill(examples=slots['[EXAMPLE]'],test=True)
        return super().fill(slots)


class Concat_Ins_behavior_Template_Llama(Template):
    def __init__(self) -> None:
        super().__init__()
        self.template = \
'''[INS]

#### Examples
Here are some examples to help you thinking.

[EXAMPLE]
[ANSWER]
'''
        self.slots = ['[INS]','[EXAMPLE]','[ANSWER]']
    def fill(self, slots):
        example_template = ExampleTemplate()
        slots['[EXAMPLE]'] = example_template.fill(examples=slots['[EXAMPLE]'],test=True)
        return super().fill(slots)




