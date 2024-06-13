from transformers import AutoModelForCausalLM, AutoTokenizer


class BLOOMZ():
    def __init__(self, name="bigscience/bloomz-560m", qa_prefix=True):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype="auto", device_map="auto"
        )
        self.qa_prefix = qa_prefix
        self.reset()

    def reset(self, instruction=""):
        self.init_context = instruction
        self.history = []

    @property
    def context(self):
        return self.init_context + self.rebuild_context(self.history)

    def __call__(self, prompt, max_new_tokens=200):
        if self.qa_prefix:
            full_prompt = self.context + f"Q: {prompt}\n\nA: "
        else:
            full_prompt = self.context + f"{prompt}\n\n"

        input_ = self.tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(input_, max_new_tokens=max_new_tokens)
        output = self.tokenizer.decode(output[0, input_.shape[1]:]).rstrip("</s>")

        self.history.append((prompt, output))
        return output

    def rebuild_context(self, qa_list):
        context = ""
        for q, a in qa_list:
            if self.qa_prefix:
                if q is not None:
                    context += f"Q: {q}\n\n"
                if a is not None:
                    context += f"A: {a}\n\n"
            else:
                if q is not None:
                    context += f"{q}\n\n"
                if a is not None:
                    context += f"{a}\n\n"

        return context

    def revoke(self, n=1):
        assert 0 <= n and n <= len(self.history)
        if n == 0:
            return
        self.history = self.history[:-n]

    def force(self, new_reply):
        self.history[-1] = (self.history[-1][0], new_reply)
