from transformers import LlamaForCausalLM, LlamaTokenizer

from .build import MODELS


@MODELS.register()
class Llama():
    def __init__(self, tokenizer_path, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
        self.reset()

    def reset(self, instruction=""):
        self.init_context = self.format_instruction(instruction)

        # `self.history` format:
        # [
        #     (Q1, A1), # not forced
        #     (Q2, A2, A2_old, A2_older, ..)  # forced
        # ],
        self.history = []

    @property
    def context(self):
        return self.init_context + self.rebuild_context(self.history)

    def __call__(self, prompt, max_new_tokens=20):
        full_prompt = self.context + f"### Input:\n{prompt}\n\n" + "### Response:\n"

        input_ = self.tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(input_, max_new_tokens=max_new_tokens)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(full_prompt):]

        self.history.append((prompt, output))
        return output

    def format_instruction(self, instruction):
        return "Below is an instruction that describes a task, " \
               "paired with an input that provides further context. " \
               "Write a response that appropriately completes the request.\n\n" \
               "### Instruction:\n" \
               f"{instruction}\n\n"

    def rebuild_context(self, qa_list):
        context = ""
        for qa in qa_list:
            q, a = qa[:2]
            if q is not None:
                context += f"### Input:\n{q}\n\n"
            if a is not None:
                context += f"### Response:\n{a}\n\n"

        return context

    def add_history(self, qa_lists):
        for qa_list in qa_lists:
            self.history += qa_list

    def revoke(self, n=1):
        assert 0 <= n and n <= len(self.history)
        if n == 0:
            return
        self.history = self.history[:-n]

    def force(self, new_reply):
        assert isinstance(new_reply, str)
        self.history[-1] = (self.history[-1][0], new_reply, *self.history[-1][1:])
