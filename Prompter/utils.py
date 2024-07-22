import numpy as np

class InsRecord():

    def __init__(self, Record=None):
        self.step_record = {}
        self.step_test_record = {}
        self.step_ops_record={}
        if Record is not None:
            self.step_record[0] = Record    

    
    def add_Ins_Record(self,step,Record):
        if step not in self.step_record:
            self.step_record[step] = Record
        else:
            self.step_record[step].extend(Record)
            
    def add_Test_Record(self,step,Record):
        #if step not in self.step_test_record:
        self.step_test_record[step] = Record
        # else:
        #     self.step_test_record[step].extend(Record)
    
    def add_Ops_Record(self,step,dialog_history):
        if step not in self.step_ops_record:
            self.step_ops_record[step] = dialog_history
        else:
            self.step_ops_record[step].extend(dialog_history)


    def reset_Record(self):
        self.step_record = {}
       
                # self.prompts = []
        # self.scores = []

import re 

def check_template_kwargs(template,kwargs):
    valid = True
    invalid_params = []
    for slot_name in template.slots:
        if slot_name == "[INS]":
            continue
        else:
            if slot_name not in kwargs:
                invalid_params.append(slot_name)
                valid = False
    if valid :
        return True
    else:
        for invalid_param in invalid_params:
            raise ValueError(
                f"{type(template)} need the params ``{invalid_param}``"
            )
        
def format_feedback(feedback):
    # pattern = r'<START>(.*?)<END>'
    # feedback_parse = re.findall(pattern,feedback[0])
    pattern = re.compile(r'<START>(.*?)<END>', re.DOTALL)
    feedback_parse = pattern.findall(feedback[0])
    delimiter = "\n"
    feedback_string = ""

    for fd_ind,feedback_content in enumerate(feedback_parse):
        feedback_string += feedback_content
        if fd_ind != len(feedback_parse)-1:
            feedback_string += delimiter   
    if not feedback_string:
        feedback_string = feedback[0]
    return feedback_string

def extract_text_format(input_texts,pattern = r'<INS>(.*?)</INS>'):
    if not isinstance(input_texts,list):
        input_texts = [input_texts]
    extract_texts = []
    for input_text in input_texts:
        matches = re.findall(pattern,input_text,re.DOTALL)
        if matches:
            extract_texts.extend(matches)
        else:
            extract_texts.append(input_text)
    return extract_texts
    

def extract_instruction(input_texts):

    inner_pattern = r'Instruction \d+:\s*(.*)'
    outter_pattern = r'<START>(.*?)<END>'
    all_instructions = []
    if not isinstance(input_texts,list):
        input_texts = [input_texts]
    for input_text in input_texts:
    
        matches = re.findall(outter_pattern, input_text, re.DOTALL)

        if matches:
            temp_result = [match.strip() for match in matches]
            result = []
            for item in temp_result:
                inner_unwrap = re.findall(inner_pattern,item)
                if inner_unwrap:
                    result.extend(inner_unwrap)
                else:
                    result.append(item)

        else:
            result = [input_text]
        all_instructions.extend(result)

    return all_instructions
