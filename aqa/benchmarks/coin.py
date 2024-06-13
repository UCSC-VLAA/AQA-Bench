import random

from loguru import logger

from .binary_search import BinarySearch
from .build import BENCHMARKS


@BENCHMARKS.register()
class Coin(BinarySearch):
    def __init__(
        self, min=0, max=100,
        format_tolerant=True, max_retry=0, max_step=None,
        verbose=True, output_dir=None, save_period=-1
    ):
        super(Coin, self).__init__(
            min, max, format_tolerant, max_retry, max_step, verbose, output_dir, save_period,
        )

    def reset(self, test_case=None):
        super(Coin, self).reset(test_case)

        if test_case is None:
            logger.info("Generating random number.")
            self._target = random.randint(self.min, self.max)
        else:
            logger.info("Using pre-generated random number.")
            self._target = test_case["target"]
            assert self.min <= self._target and self._target <= self.max, self._target

    @property
    def default_instruction(self):
        return "You're in a hidden temple where an old witch sits with a chest of gold. " \
               "The witch promises to reward you with gold coins, the amount hidden within the chest ranging from {} to {}. " \
               "To claim your prize, you must correctly guess the exact number of gold coins in the chest. " \
               "After each guess, the witch will hint if the actual amount is higher or lower than your guess. " \
               "Use these clues to adjust your guess accordingly. " \
               "Try as few times as you can. " \
               "You can only reply with a integer number between {} and {}." \
               .format(self.min, self.max, self.min, self.max)

    def _get_prompt(self, guess):
        if guess < self._target:
            return f"The true number of gold coins is bigger than {guess}."
        if guess > self._target:
            return f"The true number of gold coins is smaller than {guess}."

        return f"Right answer. The true number of gold coins is equal to {guess}."

    def _get_prompt_when_weak_tg(self):
        return "Your output is not optimal. You should follow binary-search algorithm. " \
               f"You can only reply with a integer number between {self.min} and {self.max}. " \
               "Try again."
