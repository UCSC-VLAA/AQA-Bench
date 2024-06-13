import random
import re

from loguru import logger

from aqa.models import BSModel
from aqa.utils import Invalid, FormatInvalid, ValueInvalid

from .benchmark import Benchmark
from .build import BENCHMARKS


@BENCHMARKS.register()
class BinarySearch(Benchmark):
    def __init__(
        self, min=0, max=100,
        format_tolerant=True, max_retry=0, max_step=None,
        verbose=True, output_dir=None, save_period=-1
    ):
        super(BinarySearch, self).__init__(
            format_tolerant, max_retry, max_step, verbose, output_dir, save_period
        )
        assert min <= max
        self.min = min
        self.max = max
        self.teacher = BSModel(min, max)

    def reset(self, test_case=None):
        super(BinarySearch, self).reset(test_case)

        if test_case is None:
            logger.info("Generating random number.")
            self._target = random.randint(self.min, self.max)
        else:
            logger.info("Using pre-generated random number.")
            self._target = test_case["target"]
            assert self.min <= self._target and self._target <= self.max, self._target

    @property
    def default_instruction(self):
        return "You are required to guess the random number which I have just picked between {} and {}. " \
               "I will only tell you whether the true number is bigger or lower than your guess. " \
               "Adjust your guess according to my response. " \
               "Try as few times as you can. " \
               "You can only reply with a integer number between {} and {}." \
               .format(self.min, self.max, self.min, self.max)

    def _get_prompt(self, guess):
        if guess < self._target:
            return f"The true number is bigger than {guess}."
        if guess > self._target:
            return f"The true number is smaller than {guess}."

        return f"Right answer. The true number is equal to {guess}."

    def _get_prompt_when_weak_tg(self):
        return "Your output is not optimal. You should follow binary-search algorithm. " \
              f"You can only reply with a integer number between {self.min} and {self.max}. " \
               "Try again."

    def _refresh_teacher_qa(self):
        super(BinarySearch, self)._refresh_teacher_qa()

        guess = None
        prompt = "START"

        while guess != self._target:
            guess = int(self.teacher(prompt))
            self._teacher_qa_list.append((prompt, guess))

            prompt = self._get_prompt(guess)

        self._teacher_qa_list.append((prompt, None))

    def _extract_answer(self, reply):
        # parse reply from model and return the formatted answer
        # return an `Invalid` if failed to do so
        if self.format_tolerant:
            nums = re.findall(r'\d+', reply)
            if not len(nums):
                return FormatInvalid(reply)

            guess = int(nums[0])

            if guess < self.min or guess > self.max:
                return ValueInvalid(guess)
            return guess

        try:
            guess = int(reply)

            if guess < self.min or guess > self.max:
                return ValueInvalid(guess)
            return guess
        except ValueError:
            return FormatInvalid(guess)

    def _check_algo(self, answer, answer_list):
        '''
        Check whether `answer` follows the binary-search algorithm
        Will assume the previous answer in `answer_list` already follow the binary-search algorithm

        Return
        - boolean: if selected interface follows binary-search
        '''
        left = max(
            [self.min] + [pre_answer for pre_answer in answer_list if pre_answer < self._target]
        )
        right = min(
            [self.max + 1] + [pre_answer for pre_answer in answer_list if pre_answer > self._target]
        )
        mid = (left + right) // 2

        return answer == mid

    def _calc_err(self, guess, target):
        # calculate the error between a single guess and target
        if isinstance(guess, Invalid):
            return 1

        guess = int(guess)
        target = int(target)
        return abs(guess - target) / (self.max - self.min)

    def calc_metric_tf(self, answer_list, teacher_answer_list):
        assert len(answer_list) == len(teacher_answer_list)

        if not len(answer_list):
            return {
                "avg_err": 1.0,
                "sum_err": 1.0,
                "min_err": 1.0,
                "acc": 0.0
            }

        err_list = [
            self._calc_err(answer, self._target) for answer in answer_list
        ]

        metrics = {
            "avg_err": sum(err_list) / len(err_list),
            "sum_err": sum(err_list),
            "min_err": min(err_list),
        }

        cnt = 0
        for idx, answer in enumerate(answer_list):
            if self._check_algo(answer, teacher_answer_list[:idx]):
                cnt += 1

        metrics["acc"] = cnt / len(answer_list)

        return metrics

    def calc_metric_no_tf(self, answer_list):
        if not len(answer_list):
            return {
                "avg_err": 1.0,
                "sum_err": 1.0,
                "min_err": 1.0,
                "acc": 0.0
            }

        err_list = [
            self._calc_err(answer, self._target) for answer in answer_list
        ]

        metrics = {
            "avg_err": sum(err_list) / len(err_list),
            "sum_err": sum(err_list),
            "min_err": min(err_list),
        }

        highest_cnt = 0
        for i, answer in enumerate(answer_list):
            if not self._check_algo(answer, answer_list[:i]):
                break
            highest_cnt += 1
        metrics["acc"] = highest_cnt / len(answer_list)

        return metrics

    def _test_no_tf(self, model, weak_tg_chances=0):
        # test one time without teacher forcing
        answer = None
        answer_list = []
        prompt = "START"

        retry_cnt = 0
        # for weak tg
        weak_tg_cnt = 0
        do_algo_check = weak_tg_chances > 0

        while (
            answer != self._target
            # stop when reaching `self.max_step`
            and (self.max_step is None or len(answer_list) < self.max_step)
            # stop when reaching `self.max_retry`
            and retry_cnt < (self.max_retry + 1)
        ):
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)
            self.dialog_logger.info(A=reply)

            answer = self._extract_answer(reply)

            # if `reply` is formatted in `_extract_answer`, force the new reply
            if not isinstance(answer, FormatInvalid) \
               and str(getattr(answer, "output", answer)) != reply:
                assert self.format_tolerant
                formatted = getattr(answer, "output", answer)
                assert isinstance(formatted, int)
                logger.info(f"Format tolerance enabled, force the model reply to {formatted}.")
                model.force(str(formatted))

            if isinstance(answer, Invalid):
                prompt = "Invalid reply. You can only reply with a integer number between " \
                        f"{self.min} and {self.max}. Try again."
                retry_cnt += 1
                continue

            if do_algo_check:
                if weak_tg_cnt < weak_tg_chances:
                    is_following_algo = self._check_algo(answer, answer_list)
                    if not is_following_algo:
                        weak_tg_cnt += 1
                        prompt = self._get_prompt_when_weak_tg()
                        continue

                weak_tg_cnt = 0

                do_algo_check = is_following_algo

            prompt = self._get_prompt(answer)
            answer_list.append(answer)
            retry_cnt = 0

        self.dialog_logger.info(Q=prompt)

        if isinstance(answer, Invalid):
            answer_list.append(answer)  # save the last invalid
            logger.info("Max retry times reached, stop interaction now.")
        elif answer != self._target:  # target not achieved
            logger.info("Max steps reached, stop the interaction now.")

        return answer_list

    def _test_tf(self, model):
        # test one time with teacher forcing
        answer = None
        answer_list = []
        teacher_answer_list = []

        self._refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_answer in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)

            model.force(str(teacher_answer))
            self.dialog_logger.info(A=reply, T=teacher_answer)

            answer = self._extract_answer(reply)

            answer_list.append(answer)
            teacher_answer_list.append(teacher_answer)

        self.dialog_logger.info(Q=self._teacher_qa_list[-1][0])

        return answer_list, teacher_answer_list

    def naive_test(self, model, teacher_forcing=False, weak_tg_chances=0, instruction=None):
        logger.info("Target number: {}".format(self._target))

        if teacher_forcing:
            answer_list, teacher_answer_list = self._test_tf(model)
            metric = self.calc_metric_tf(answer_list, teacher_answer_list)
        else:
            teacher_answer_list = []
            answer_list = self._test_no_tf(model, weak_tg_chances)
            metric = self.calc_metric_no_tf(answer_list)

        result = self._get_result(
            metric, answer_list, teacher_answer_list,
            model.history, teacher_forcing, instruction
        )

        return metric, result

    def _get_result(
        self, metric, answer_list, teacher_answer_list,
        model_history, teacher_forcing, instruction=None
    ):
        result = super(BinarySearch, self)._get_result(
            metric, answer_list, teacher_answer_list, model_history, teacher_forcing, instruction
        )

        result["env"].update(
            min=self.min,
            max=self.max,
            target=self._target,
        )

        return result

    def _pack_results(self, single_results, teacher_forcing_mode):
        metric, full_result = super(BinarySearch, self)._pack_results(
            single_results, teacher_forcing_mode
        )

        full_result["env"]["min"] = self.min
        full_result["env"]["max"] = self.max

        return metric, full_result
