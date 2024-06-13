import google.generativeai as genai
from loguru import logger
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from .build import MODELS


def _log_when_fail(retry_state):
    logger.info(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )


@MODELS.register()
class Gemini():
    def __init__(self, api_key, sleep_sec):
        self.sleep_sec = sleep_sec

        if isinstance(api_key, str):
            self.api_keys = [api_key]
        else:
            self.api_keys = api_key

        # Set up the model
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 128,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]

        self.client = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        self.reset()

    def reset(self, instruction=None):
        self.instruction = instruction
        self.history = []

    def _build_conv(self):
        if self.instruction is None:
            messages = []
        else:
            messages = [
                {
                    "role": "user",
                    "parts": [self.instruction]
                },
                {
                    "role": "model",
                    "parts": [""]
                },
            ]

        for qa in self.history:
            q, a = qa[:2]
            messages.append({"role": "user", "parts": [q]})
            messages.append({"role": "model", "parts": [a]})

        conv = self.client.start_chat(history=messages)
        return conv

    def __call__(self, prompt):
        time.sleep(self.sleep_sec)
        key = self.api_keys.pop(0)
        genai.configure(api_key=key)
        self.api_keys.append(key)

        conv = self._build_conv()

        retry(
            wait=wait_random_exponential(multiplier=1, max=1000),
            stop=stop_after_attempt(15),
            before_sleep=_log_when_fail
        )(conv.send_message)(prompt)

        result = conv.last.text
        self.history.append((prompt, result))

        return result

    def add_history(self, qa_lists):
        for qa_list in qa_lists:
            self.history += qa_list

    def revoke(self, n=1):
        assert 0 <= n and n <= len(self.history)
        if n == 0:
            return
        self.history = self.history[:-n]

    def force(self, new_reply):
        self.history[-1] = (self.history[-1][0], new_reply, *self.history[-1][1:])
