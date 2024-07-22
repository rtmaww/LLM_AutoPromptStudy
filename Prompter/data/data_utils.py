import random
import os
import json
SEED=0

def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return inputs, outputs


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    random.seed(SEED)
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)
def get_task_desc(task, data_cty):
    base_path = f'AutoPrompter/Prompter/data/{data_cty}'
    
    induce_data_path = os.path.join(base_path, 'raw/induce/')
    base_path = induce_data_path 
        
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    task_desc = data['task_desc']
    return task_desc

def load_data(type, task, data_cty):
    base_path = f'AutoPrompter/Prompter/data/{data_cty}'
    
    induce_data_path = os.path.join(base_path, 'raw/induce/')
    eval_data_path = os.path.join(base_path, 'raw/execute/')
    ini_data_path =  os.path.join(base_path, 'raw/ini/')

    # Get a list of tasks (by looking at the names of the files in the induced directory)
    tasks = [f.split('.')[0] for f in os.listdir(induce_data_path)]

    if type == "eval":
        base_path = induce_data_path
    elif type == "test":
        base_path = eval_data_path
    else:
        base_path = ini_data_path

    #base_path = induce_data_path if type == 'induce' else eval_data_path
        
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []
    for i in range(num_examples):
        data = examples[str(i + 1)]
        if data_cty in ['bigbench-ii']:
            input_, output_ = data['input'], data['output']
        else:
        
            if task == 'cause_and_effect':
                cause, effect = data['cause'], data['effect']
                # Pick an order randomly
                if random.random() < 0.5:
                    input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
                else:
                    input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
                output_ = [cause]
            elif task == 'common_concept':
                items = data['items']
                # Make comma separated list of items
                input_ = ', '.join(items[:-1])
                output_ = data['all_common_concepts']
            elif task == 'rhymes':
                input_, output_ = data['input'], data['other_rhymes']
            elif 'translation' in task:
                input_, output_ = data['input'], data['possible_translations']
            else:
                input_, output_ = data['input'], [data['output']]
        inputs.append(input_)
        outputs.append(output_)
    return inputs, outputs
