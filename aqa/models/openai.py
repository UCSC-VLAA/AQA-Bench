from loguru import logger
import openai
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
class OpenAI():
    def __init__(self, model_name, api_key, api_version, end_point, sleep_sec=0.5):
        self.model_name = model_name
        self.sleep_sec = sleep_sec
        self.client = openai.AzureOpenAI(
            azure_endpoint=end_point,
            api_key=api_key,
            api_version=api_version
        )
        self.completion_func = retry(
            wait=wait_random_exponential(min=1, max=5),
            stop=stop_after_attempt(15),
            before_sleep=_log_when_fail
        )(self.client.chat.completions.create)

        self.reset()

    def reset(self, instruction="You are a chatbot"):
        self.messages = [{
            "role": "system",
            "content": [{"type": "text", "text": instruction},]
        }]
        self.history = []

    def __call__(self, prompt):
        time.sleep(self.sleep_sec)
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt},],
        })

        response = self.completion_func(
            model=self.model_name,
            messages=self.messages,
            max_tokens=128,
            temperature=0.0,
            seed=42
        )

        result = ""
        for choice in response.choices:
            if choice.message.content is not None:
                result += choice.message.content

        self.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": result},],
        })
        self.history.append((prompt, result))

        return result

    def rebuild_context(self, qa_list):
        context = ""
        for qa in qa_list:
            q, a = qa[:2]
            if q is not None:
                context += f"user: {q}\n\n"
            if a is not None:
                context += f"assistant: {a}\n\n"

        return context

    def add_history(self, qa_lists):
        for qa_list in qa_lists:
            self.history += qa_list

    def revoke(self, n=1):
        assert 0 <= n and n <= len(self.history)
        if n == 0:
            return
        self.history = self.history[:-n]
        self.messages = self.messages[:-n]

    def force(self, new_reply):
        self.history[-1] = (self.history[-1][0], new_reply, *self.history[-1][1:])
        self.messages[-1]["content"][0]["text"] = new_reply
