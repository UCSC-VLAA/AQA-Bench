from typing import Optional

import torch

from fastchat.conversation import get_conv_template
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)

from fastchat.modules.awq import AWQConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.utils import get_context_length

from .build import MODELS


@MODELS.register()
class FastChatModel():
    def __init__(
        self,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        dtype: Optional[torch.dtype],
        load_8bit: bool,
        cpu_offloading: bool,
        conv_template: Optional[str],
        conv_system_msg: Optional[str],
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        revision: str = "main",
        judge_sent_end: bool = True,
    ):
        self.conv_template = conv_template
        self.model_path = model_path
        self.conv_system_msg = conv_system_msg
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.judge_sent_end = judge_sent_end

        if num_gpus == "all":
            num_gpus = torch.cuda.device_count()

        self.model, self.tokenizer = load_model(
            model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=GptqConfig(ckpt=model_path, wbits=16, groupsize=-1, act_order=False),
            awq_config=AWQConfig(ckpt=model_path, wbits=16, groupsize=-1),
            exllama_config=exllama_config,
            xft_config=xft_config,
            revision=revision,
            debug=False,
        )

        self.generate_stream_func = get_generate_stream_function(self.model, model_path)

        model_type = str(type(self.model)).lower()
        is_t5 = "t5" in model_type
        self.is_codet5p = "codet5p" in model_type

        self.repetition_penalty = repetition_penalty

        # Hardcode T5's default repetition penalty to be 1.2
        if is_t5 and self.repetition_penalty == 1.0:
            self.repetition_penalty = 1.2

        # Set context length
        self.context_len = get_context_length(self.model.config)

    def reset(self, instruction=""):
        if self.conv_template:
            self.conv = get_conv_template(self.conv_template)
        else:
            self.conv = get_conversation_template(self.model_path)
        if self.conv_system_msg is not None:
            self.conv.set_system_message(self.conv_system_msg)

        if instruction != "":
            self.conv.set_system_message(instruction)

        # `self.history` format:
        # [
        #     (Q1, A1), # not forced
        #     (Q2, A2, A2_old, A2_older, ..)  # forced
        # ],
        self.history = []

    def __call__(self, inp):
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        if self.is_codet5p:  # codet5p is a code completion model.
            prompt = inp

        gen_params = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "stop": self.conv.stop_str,
            "stop_token_ids": self.conv.stop_token_ids,
            "echo": False,
        }

        output_stream = self.generate_stream_func(
                self.model,
                self.tokenizer,
                gen_params,
                self.device,
                context_len=self.context_len,
                judge_sent_end=self.judge_sent_end,
        )

        output = list([i for i in output_stream])[-1]["text"].strip()

        self.conv.update_last_message(output)
        self.history.append((inp, output))

        return output

    def add_history(self, qa_lists):
        for qa_list in qa_lists:
            self.history += qa_list
            for q, a in qa_list:
                self.conv.append_message(self.conv.roles[0], q)
                self.conv.append_message(self.conv.roles[1], a)

    def revoke(self, n=1):
        assert 0 <= n and n <= len(self.history)
        if n == 0:
            return
        self.history = self.history[:-n]
        self.conv.messages = self.conv.messages[:-2 * n]

    def force(self, new_reply):
        assert isinstance(new_reply, str)
        self.history[-1] = (self.history[-1][0], new_reply, *self.history[-1][1:])
        self.conv.update_last_message(new_reply)
