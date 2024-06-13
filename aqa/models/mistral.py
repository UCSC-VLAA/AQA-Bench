import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .build import MODELS


@MODELS.register()
class Mistral():
    def __init__(self, model_id, max_new_tokens=128):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if "Mixtral" in self.model_id:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.reset()

    def reset(self, instruction=None):
        self.instruction = instruction

        # `self.history` format:
        # [
        #     (Q1, A1), # not forced
        #     (Q2, A2, A2_old, A2_older, ..)  # forced
        # ],
        self.history = []

    def __call__(self, prompt, max_new_tokens=20):
        input_ = self.get_encoded(prompt).to("cuda")

        output = self.model.generate(input_, max_new_tokens=max_new_tokens, do_sample=False)
        output = output[:, input_.shape[1]:]
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        self.history.append((prompt, output))
        return output

    def get_encoded(self, inp):
        messages = []

        for qa in self.history:
            q, a = qa[0], qa[1]
            if a is None:
                a = ""
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": inp})

        if self.instruction is not None:
            # messages[0]["content"] = f"<<SYS>>\n{inp}\n<</SYS>>\n\n" + messages[0]["content"]
            messages.insert(0, {"role": "user", "content": self.instruction})
            messages.insert(1, {"role": "assistant", "content": ""})

        # if "Mixtral" not in self.model_id:
        return self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        # text = "<s>"
        # for q, a in zip(messages[::2], messages[1::2]):
        #     text += f" [INST] {q['content']} [/INST] {a['content']}</s>"

        # text += f" [INST] {messages[-1]['content']} [/INST] "
        # return self.tokenizer.encode(text, return_tensors="pt")

    def _tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

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
