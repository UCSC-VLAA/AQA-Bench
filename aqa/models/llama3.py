import torch
from transformers import pipeline

from .build import MODELS


@MODELS.register()
class Llama3():
    def __init__(self, model_id, max_new_tokens=32):
        self.model_id = model_id
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
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

    def __call__(self, prompt):
        messages = self.get_messages(prompt)

        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
        )

        output = outputs[0]["generated_text"][-1]["content"]
        self.history.append((prompt, output))

        return output

    def get_messages(self, inp):
        messages = []

        for qa in self.history:
            q, a = qa[0], qa[1]
            if a is None:
                a = ""
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": inp})

        if self.instruction is not None:
            messages.insert(0, {"role": "system", "content": self.instruction})

        return messages

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
