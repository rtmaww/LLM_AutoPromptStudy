'''
    class used to wrap message to adapt different LLMs.
'''
class PromptBuilder(object):
    def __init__(self) -> None:
        pass

    def build_prompt(self,message):
        pass


class LlamaPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        super().__init__()

    def build_prompt(self, messages):
        startPrompt = "<s>[INST] "
        endPrompt = " [/INST]"
        conversation = []
        for index, message in enumerate(messages):
            if message["role"] == "system" and index == 0:
                conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
            elif message["role"] == "user":
                conversation.append(message["content"].strip())
            else:
                conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

        return startPrompt + "".join(conversation) + endPrompt

class VicunaPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        super().__init__()

    def build_prompt(self, messages):
       
        conversation = []
        for index, message in enumerate(messages):
            if message["role"] == "system":
                conversation.append(f"{message['content']}\n")
            elif message["role"] == "user":
                conversation.append(f"USER: {message['content'].strip()}\n")
            elif message["role"] == "assistant":
                conversation.append(f"ASSISTANT: {message['content'].strip()}</s>\n")

        return  "".join(conversation)
    