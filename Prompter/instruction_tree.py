from graphviz import Digraph
from AutoPrompter import Random_Hint
import json
from args import parse_args
import re
import os
file_step = 1
def format_string_length(text, max_words_per_line=20):
    words = text.split(" ")
    formatted_text = ""
    words_count = 0

    for word in words:
        if '\n' in word:
            words_count = 0
        formatted_text += word + " "
        words_count += 1

        if words_count >= max_words_per_line:
            formatted_text += "\n"
            words_count = 0

    return formatted_text.strip()

def format_output(content):
    delimiter = "\n"
    if isinstance(content,str):
        content = content.replace("\\n","\n")
        return format_string_length(content)
    #if isinstance(content,list):
    #    format_string = ""
    #    for ind,item in enumerate(content):
    #        format_string += format_string_length(str(item))
    #        if ind != len(content)-1:
    #            format_string += delimiter
    #    return format_string
    if isinstance(content,list):
        if isinstance(content[0],list):
            format_string = ""
            _input,llm_output,_output,llm_format_output = content[0],content[1],content[2],content[3]
            length = len(_input)
            for ind in range(length):
                format_string += f"Input:{format_string_length(str(_input[ind]))}\n\nOutput:{format_string_length(str(_output[ind]))}\n\nllm_output:{format_string_length(str(llm_output[ind]))}\n\nllm_format_output:{format_string_length(str(llm_format_output[ind]))}\n\n"
                if ind != length-1:
                    format_string += delimiter
        else:
            format_string = ""
            for ind,item in enumerate(content):
                format_string += format_string_length(str(item))
                if ind != len(content)-1:
                    format_string += delimiter

        return format_string
    return format_string_length(str(content))

def chunk_result_data(data,length=2):
    if isinstance(data,str):
        return
    input,output,LLM_output,format_output = data
    if len(data[0]) < length:
        return data
    for i in range(len(data)):
        data[i] = data[i][:length]
    return data

class TreeNode:
    def __init__(self, 
                 father_id = "",
                 instruction="",
                 correct_dev="",
                 error_dev="",
                 feedback="",
                 eval_score=0,
                 test_score=0,
                 correct_test="",
                 error_test="",
                 ):
        self.children = []
        self.father_node_id = father_id
        self.data = {
                    "instruction":instruction,
                    "eval_score":eval_score,
                    "test_score":test_score,
                    "feedback":feedback,
                    "correct_dev":correct_dev,
                    "error_dev":error_dev,
                    "correct_test":correct_test,
                    "error_test":error_test
                    }
        self.state = True
        
        

    def add_child(self, child_node):
        if isinstance(child_node,list):
            self.children.extend(child_node)
        else:
            self.children.append(child_node)
    def reset_child(self):
        self.children = []

class Instruction_Tree():
    def __init__(self,task="",config=None):
        self.root = TreeNode(instruction="instruction_style",father_id=-1)
        self.task = task
        self.path = f"AutoPrompter/initial_step{file_step}_layer_wise_Llama_result/{self.task}"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if config is not None:
            self.print_config = config
        else:
            self.print_config ={
                    "instruction":True,
                    "eval_score":True,
                    "test_score":False,
                    "feedback":False,
                    "correct_dev":False,
                    "error_dev":False,
                    "correct_test":False,
                    "error_test":False
        }
    def get_root(self):
        return self.root
    def build_tree_from_nested_list(self,arr):
        if not arr:
            return None

        cur_level_ind = 0
        queue = [([self.root], arr[cur_level_ind])]
        
        while queue:
            nodes, current_level = queue.pop(0)
            assert(len(nodes)==len(current_level))
            child_nodes_total = []
            for parent,sublist in zip(nodes,current_level):
                if len(sublist)==0:
                    continue
                parent.add_child(sublist)
                child_nodes_total.extend(sublist)
            cur_level_ind += 1
            if cur_level_ind < len(arr):
                queue.append((child_nodes_total, arr[cur_level_ind]))  # Skip the first element and process the rest of the sublist

        

    def get_node_label(self,node):
    
        label = ""
        for k,v in self.print_config.items():
            if v:
                label += f"{k}:{format_output(node.data[k])}\n"
        
        return label
    def print_tree(self, comment=""):

        graph = self._print_tree(self.root)
        output_path = f"{self.path}/{self.task}_instruction_tree_{comment}" if comment else f"{self.path}/{self.task}_instruction_tree"
        graph.render(filename=output_path, view=False)

    def _print_tree(self,node, graph=None):
        if graph is None:
            graph = Digraph(comment="Tree")
       
        graph.node(str(id(node)), label=self.get_node_label(node),  shape='box',)

        
        for child in node.children:
            graph.edge(str(id(node)), str(id(child)))
            self._print_tree(child, graph)

        return graph


            
    def tree_to_dict(self,node):
        
        data = {"data": node.data, "children": []}
        for child in node.children:
            data["children"].append(self.tree_to_dict(child))
        return data

    def save_tree_to_json(self, comment=""):
    
        data = self.tree_to_dict(self.root)
        if comment:
            save_file = f"tree_{comment}.json"
        else:
            save_file = "tree.json"
        with open(f"{self.path}/{save_file}", "w") as file:
            json.dump(data, file)
    
    def get_all_step_instructions(self):
        all_instructions = {}
        queue = [[self.root]]
        cur_level_ind = 0
        while queue:
            node_list = queue.pop(0)
            cur_level_child_nodes = []
            cur_level_instructions = []
            for node in node_list:
                cur_level_instructions.append(node.data['instruction'])
                cur_level_child_nodes.extend(node.children)
            
            all_instructions[cur_level_ind] = cur_level_instructions
            cur_level_ind += 1
            if len(cur_level_child_nodes) > 0:
                queue.append(cur_level_child_nodes)
        return all_instructions
    
    def get_all_step_feedbacks(self):
        all_feedbacks = {}
        queue = [[self.root]]
        cur_level_ind = 0
        while queue:
            node_list = queue.pop(0)
            cur_level_child_nodes = []
            cur_level_feedbacks = []
            for node in node_list:
                    
                cur_level_feedbacks.append(node.data['feedback'])
                cur_level_child_nodes.extend(node.children)
            
            all_feedbacks[cur_level_ind] = cur_level_feedbacks
            cur_level_ind += 1
            if len(cur_level_child_nodes) > 0:
                queue.append(cur_level_child_nodes)
        return all_feedbacks
    
    
    def get_test_step_instructions(self):
        all_instructions = {}
        queue = [[self.root]]
        cur_level_ind = 0
        while queue:
            node_list = queue.pop(0)
            cur_level_child_nodes = []
            cur_level_instructions = []
            for node in node_list:
                if isinstance(node.data["test_score"],list):
                    all_instructions[node.data['instruction']]=node.data["test_score"][0]
                cur_level_child_nodes.extend(node.children)
            
        
            cur_level_ind += 1
            if len(cur_level_child_nodes) > 0:
                queue.append(cur_level_child_nodes)
        return all_instructions
    
    def get_instructions(self):
        all_instructions = {}
        queue = [[self.root]]
        cur_level_ind = 0
        while queue:
            node_list = queue.pop(0)
            cur_level_child_nodes = []
            cur_level_instructions = []
            cur_level_ins_scores = []
            for node in node_list:
            
                cur_level_instructions.append({"instruction":node.data['instruction'],"score":node.data['eval_score']})
                cur_level_child_nodes.extend(node.children)
                
            all_instructions[cur_level_ind] = cur_level_instructions
            
            cur_level_ind += 1
            
            if len(cur_level_child_nodes) == 0:
                return all_instructions
            else:
                queue.append(cur_level_child_nodes)
    def dict_to_tree(self,data):
        length = len(data["data"]["error_dev"])
        if length > 0:
            store = data["data"]["error_dev"][0][:5]
        node = TreeNode(
            instruction=data["data"]["instruction"],
            eval_score=data["data"]["eval_score"],
            test_score=data["data"]["test_score"],
            feedback=data["data"]["feedback"],
            correct_dev=chunk_result_data(data["data"]["correct_dev"]),
            error_dev=chunk_result_data(data["data"]["error_dev"]),
            correct_test=chunk_result_data(data["data"]["correct_test"]),
            error_test=chunk_result_data(data["data"]["error_test"])
            )
        for child_data in data["children"]:
            child_node = self.dict_to_tree(child_data)
            node.add_child(child_node)
        return node

    def load_tree_from_json(self, comment="",path = None):
        if path:
            self.path = path
        if comment:
            load_file = f"tree_{comment}.json"
        else:
            load_file = f"tree.json"
        if os.path.exists(f"{self.path}/{load_file}"):
            with open(f"{self.path}/{load_file}", "r") as file:
                data = json.load(file)
            self.root = self.dict_to_tree(data)         
        else:
            self.root = None




from utils import extract_instruction
pattern_dict = {
    "APO":r"My current prompt is:(.*?)But this prompt gets the following examples",
    "APE":r'Input:(.*?)Output:',
    "Random_Hint":r'The instruction is:(.*?)Please help me:',
    "APE_Plus":r'Now I have an instruction for this task:\n"(.*?)"\nPlease generate a variation of this instruction while keeping the semantic meaning.\n\nNew instruction:'
    #"APE_Plus":r'Now I have an instruction for this task:\nInstruction:\n\n(.*?)\n\nPlease generate a variation of '
}

def split_list(candidate_list, length_list):
    result = []
    idx = 0

    for length in length_list:
        if length==0:
            result.append([])
        else:
            segment = candidate_list[idx:idx+length]
            result.append(segment)
            idx += length

    return result


                
def build_tree():

    methods = ['APE']
    tasks = ["object_counting"]

    for task in  tasks:
        for ops_method in methods:
            Optimization_dict = {}
            evaluation_record_path = f"AutoPrompter/initial_step{file_step}_layer_wise_Llama_result_helpful/{ops_method}_bigbench-ii_{task}_record.json"
            with open(evaluation_record_path,'r',encoding='utf-8') as f:
                lines = f.read()
            lines = lines.split('Optimization step')[1:]
            for ind,line in enumerate(lines):
                lines[ind] = lines[ind][2:].strip()
            all_ins_score_dict = {}
            all_ins_index_dict = {}
            total_ins_score_dict = {}
            for step,step_ins_string in enumerate(lines):
            
                instruction_dict = {}
                index_dict = {}
                score_string_list = step_ins_string.split('\t')[1:]
                ins_string_list = step_ins_string.split('\t')[:-1]
                score = ""
                for ind_,(score_string,ins_string) in enumerate(zip(score_string_list,ins_string_list)):
                    if score !="":
                        ins = ins_string.split(score)[-1]
                    else:
                        ins = ins_string
                    ins = ins.strip().strip("\"").strip("\n").strip()
                
                    
                    score = score_string.split('\n')[0]
                    total_ins_score_dict[ins] = score
                    instruction_dict[ins] = {"index":ind_,"score":score}
                    index_dict[ind_] = {"ins":ins,"score":score}
                
                all_ins_score_dict[step] = instruction_dict
                all_ins_index_dict[step] = index_dict
                

            optimize_record_path = f"AutoPrompter/initial_step{file_step}_layer_wise_Llama_result_helpful/{ops_method}_bigbench-ii_{task}_optimization.json"
            with open(optimize_record_path,'r',encoding='utf-8') as f:
                ops_record = json.load(f)


            Tree_List = [[TreeNode(instruction=ins,
                                eval_score=all_ins_score_dict[0][ins]["score"]) for ins in list(all_ins_score_dict[0].keys())]]
            index_dict = {ins:ind for ind,ins in enumerate(list(all_ins_score_dict[0].keys()))}
            Total_Length_List=[[20]]
            for step,step_ops_record in ops_record.items():

                if step_ops_record is None:
                    continue
                
                step = int(step)-1
                Optimization_dict[step] = []
                print(f'step{step}:Optimize {len(step_ops_record)} instructions!\n')
            
                level_tree_node = [[] for i in range(len(Tree_List[step]))]
                length_list = [0 for i in range(len(Tree_List[step]))]

                for ops_item_ind,ops_item in enumerate(step_ops_record):
                    
                    if ops_method == "APO":
                        cur_ops_item  = ops_item[0]
                        feedback_history = cur_ops_item[0]
                        update_history = cur_ops_item[1]
                    elif ops_method == "Random_Hint":
                        cur_ops_item  = ops_item[0]
                        feedback_history = cur_ops_item
                        update_history = cur_ops_item
                    elif  ops_method == "APE_Plus":
                        feedback_history = ops_item[0]
                        update_history = ops_item[0]
                    elif ops_method == "APE" :
                        feedback_history = ops_item
                        update_history = ops_item
                    #user query:include origin instrucion
                    feedback_query = feedback_history[1]["content"]
                    #assistant:include LLM answer
                    feedback_return_content = feedback_history[2]["content"]

                    pattern = pattern_dict[ops_method]

                    # Use re.search to find the origin instruction
                    match = re.search(pattern, feedback_query, re.DOTALL)
                    
                    if match:
                        ins_content = match.group(1).strip()
                        ins_content = ins_content.strip("\"").strip().strip("\"").strip()
                        origin_ins_dict = {"instruction":ins_content,"score":total_ins_score_dict[ins_content]}
                        
                        
                        ## if need tree ,needed this code!!
                        
                        ins_index = index_dict[ins_content]
                        if ops_method == "APO" or ops_method =="Random_Hint":
                            Tree_List[step][ins_index].data['feedback'] = feedback_return_content
                        
                        length_list[ins_index] = 5
                    
                    
                    else:
                        raise ValueError("ins not found in query.")
                    
         
                    
                    update_return = update_history[2]["content"]
                    update_instructions = extract_instruction(update_return)
                    for upd_ins in update_instructions:
                        upd_ins = upd_ins.strip().strip("\"").strip()
                        upd_score = total_ins_score_dict[upd_ins]
                        upd_ins_dict_1 = {"instruction":upd_ins,"score":upd_score}
                        Optimization_dict[step].append({"pre_ins":origin_ins_dict,"post_ins":upd_ins_dict_1,"feedback":feedback_return_content})
                        level_tree_node[ins_index].append(TreeNode(instruction=upd_ins,
                                                                   eval_score=all_ins_score_dict[step+1][upd_ins]["score"]))
                        if ops_method == "APO" and len(update_instructions)==1:
                            next_index = all_ins_score_dict[step+1][upd_ins]["index"]
                            level_tree_node[ins_index].append(TreeNode(instruction=all_ins_index_dict[step+1][next_index+1]["ins"],
                                                                       eval_score=all_ins_index_dict[step+1][next_index+1]["score"]))
                            
                    
                        if ops_method == "APE" or ops_method == "Random_Hint" or ops_method == "APE_Plus":
                        #if  ops_method == "Random_Hint":
                            next_index = all_ins_score_dict[step+1][upd_ins]["index"]
                            upd_instruction=all_ins_index_dict[step+1][next_index+1]["ins"],
                            upd_eval_score=all_ins_index_dict[step+1][next_index+1]["score"]
                            upd_ins_dict_2 = {"instruction":upd_instruction,"score":upd_eval_score}
                            Optimization_dict[step].append({"pre_ins":origin_ins_dict,"post_ins":upd_ins_dict_2,"feedback":feedback_return_content})
                        
                            for plus in range(1,5):
                                level_tree_node[ins_index].append(TreeNode(instruction=all_ins_index_dict[step+1][next_index+plus]["ins"],
                                                                       eval_score=all_ins_index_dict[step+1][next_index+plus]["score"]))
                           
                            
                        
                modify_node = []
                for item in level_tree_node:
                    modify_node.extend(item)
                index_dict={}
                for node_ind,node in enumerate(modify_node):
                    index_dict[node.data["instruction"]] = node_ind
                    

                Tree_List.append(modify_node)
                Total_Length_List.append(length_list)

            ins_tree = Instruction_Tree(task=task)
            modify_nodes=[]
            for node_list,length_list in zip(Tree_List,Total_Length_List):
                child_split_list = split_list(node_list,length_list)
                modify_nodes.append(child_split_list)
            # print(modify_nodes)

            ins_tree.build_tree_from_nested_list(arr=modify_nodes)
            ins_tree.save_tree_to_json(comment=ops_method)
            ins_tree.print_tree(comment=ops_method)
                    
            print("aa")
            with open(f"{task}_ops_record.json",'w',encoding='utf-8') as f:
                json.dump(Optimization_dict,f)
            with open(f"{task}_ops_record.txt",'w',encoding='utf-8') as f:
                for step,optimization in Optimization_dict.items():
                    f.write(f'Optimization_step{step}\n')
                    for ops_element in optimization:
                        f.write(f'origin_ins:{ops_element["pre_ins"]}\n')
                        f.write(f'ops_ins:{ops_element["post_ins"]}\n')
                        f.write(f'{ops_element["pre_ins"]["score"]}--->{ops_element["post_ins"]["score"]}\n')
                        #f.write(ops_element["feedback"])
                        f.write("\n\n")
   
    
if __name__ == '__main__':
    build_tree()
    print_config ={
                        "instruction":True,
                        "eval_score":True,
                        "test_score":True,
                        "feedback":True,
                        "correct_dev":False,
                        "error_dev":True,
                        "correct_test":True,
                        "error_test":False
            }
    args = parse_args()
    args.initial = False
    task_list = ["object_counting","snarks","navigate","question_selection"]
    eval_model_list = ["Llama","turbo"]
    methods_list = ["APO","Random_Hint"]
    for task in task_list:
        for eval_model in eval_model_list:
            for method in methods_list:
                
                args.task = task
                args.eval_model = eval_model
              
                path = f"AutoPrompter/{args.eval_model}_result/{args.eval_model}_instruction_tree/{args.task}/"
                if os.path.exists(path):
                    ins_tree = Instruction_Tree(task="disambiguation_qa_hunyuan_2",config = print_config)
                    ins_tree.load_tree_from_json(comment=method,path=path)
                    if ins_tree.root is None:
                        continue
                    all_feedbacks = ins_tree.get_all_step_feedbacks()
                    feedback_path = f"Feedback_result/{args.task}/{args.eval_model}/"
                    if not os.path.exists(feedback_path):
                        os.makedirs(feedback_path)
                    with open(f"{feedback_path}/{method}_feedbacks.txt",'w',encoding='utf-8') as f:
                        for cur_level,feedbacks in all_feedbacks.items():
                            for fdb_ind,fdb in enumerate(feedbacks):
                                new_fdbs = fdb.split('<START>')
                                if len(new_fdbs)==4:
                                    new_fdbs = new_fdbs[1:]
                                for sub_ind,sub_fdb in enumerate(new_fdbs):
                                    f.write(sub_fdb.replace('<START>','').replace('<END>','').rstrip().lstrip())
                                    f.write('\n')
            
   
   

