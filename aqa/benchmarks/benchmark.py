import abc
from loguru import logger
import json
import os.path as osp
import pickle as pkl

from aqa.utils import DialogLogger, dict_mean, InvalidEncoder, setup_logger


class Benchmark(metaclass=abc.ABCMeta):
    def __init__(
            self, format_tolerant=True, max_retry=0, max_step=None,
            verbose=True, output_dir=None, save_period=-1
    ):
        self.format_tolerant = format_tolerant
        # `max_retry` and `max_step` are only activated when not teacher forcing
        self.max_retry = max_retry
        # `max_step` will be deactivated if `max_step is None`
        self.max_step = max_step

        self.teacher = None
        self.dialog_logger = DialogLogger(order=["System", "Q", "A", "T"], enabled=verbose)
        self.test_cases = []
        self.output_dir = output_dir
        setup_logger(output=output_dir)

        # `self.save_period > 0`: save every `self.save_period` test cases
        # `self.save_period == 0`: only save the final result
        # `self.save_period < 0`: dont save results at all
        self.save_period = save_period
        assert self.save_period < 0 or self.output_dir is not None

    def load_testcases_from_file(self, path):
        logger.info(f"loading testcases from {path}")
        self.test_cases = json.load(open(path))
        assert isinstance(self.test_cases, list), self.test_cases

    def reset(self, test_case):
        self.test_case = test_case
        self.teacher.reset()
        self._teacher_qa_list = []

    def reset_model(self, model, instruction=None, example_qa_lists=None, verbose=True):
        # clear dialog history and give instruction
        # will use `self.default_instruction` if `instruction` is None
        if instruction is None:
            instruction = self.default_instruction

        model.reset(instruction)
        if verbose:
            self.dialog_logger.info(System=instruction)

        if example_qa_lists is not None:
            example_qa_lists = self._preprocess_examples(example_qa_lists)
            model.add_history(example_qa_lists)
            if verbose:
                for qa_list in example_qa_lists:
                    for q, a in qa_list:
                        self.dialog_logger.info(Q=q)
                        self.dialog_logger.info(A=a)

    def _preprocess_examples(self, qa_lists):
        example_qa_lists = []
        for qa_list in qa_lists:
            example_qa_list = [(q, str(a) if a is not None else "") for q, a in qa_list]
            example_qa_lists.append(example_qa_list)

        return example_qa_lists

    def pre_each_test(self, model, instruction=None, test_case=None, example_qa_lists=None):
        self.reset(test_case)
        # will use `self.default_instruction` if `instruction` is None
        self.reset_model(model, instruction, example_qa_lists)

    def _refresh_teacher_qa(self):
        # teacher always recieve a fresh initial prompt without previous context
        self.reset_model(self.teacher, verbose=False)
        self._teacher_qa_list = []

    def _get_result(
        self, metric, answer_list, teacher_answer_list,
        model_history, teacher_forcing, instruction=None
    ):
        result = {}
        result["metric"] = metric
        result["output"] = dict(
            answer_list=answer_list,
            teacher_answer_list=teacher_answer_list
        )
        result["env"] = dict(
            teacher_forcing=teacher_forcing,
            instruction=self.default_instruction if instruction is None else instruction
        )
        result["history"] = dict(
            model_history=model_history,
            teacher_history=self._teacher_qa_list
        )

        return result

    def _pack_results(self, single_results, teacher_forcing_mode):
        # summarize the metrics from each test run and pack the detailed results
        metrics = [result["metric"] for result in single_results]
        metric = dict_mean(metrics)

        metric = {"mean_" + k: v for k, v in metric.items()}

        full_result = {}
        full_result["metric"] = metric
        full_result["env"] = dict(
            times=len(metrics),
            teacher_forcing_mode=teacher_forcing_mode,
            default_instruction=self.default_instruction
        )
        full_result["single_results"] = single_results

        return metric, full_result

    # TODO: Deprecated
    def _independent_test(self, model, times, teacher_forcing_mode):
        teacher_forcing = teacher_forcing_mode == "l1"

        metrics = []
        single_results = []
        for i in range(times):
            metric, single_result = self.naive_test(model, teacher_forcing)

            logger.info(f"Evaluation metric #{i}: {metric}")
            metrics.append(metric)
            single_results.append(single_result)

        return self._pack_results(metrics, single_results, teacher_forcing_mode)

    # TODO: Deprecated
    def _context_kept_test(self, model, times, teacher_forcing_mode):
        # model's history will be cleared before each run
        # teacher model's history will be used as example in the intruction prompt
        # at the beginning of each run

        def get_tf_flag(i):
            # i: current run is the `i`-th run. `i` in [0, `times` - 1]
            if teacher_forcing_mode == "l2":
                return False

            if teacher_forcing_mode == "l4":
                return True

            return i < times - 1

        instruction_w_examples = self.default_instruction \
            + "\nHere are some examples (the right answer for each example is different):\n"

        metrics = []
        single_results = []
        for i in range(times):
            metric, single_result = self.naive_test(
                model, get_tf_flag(i),
                instruction=None if i == 0 else instruction_w_examples
            )

            logger.info(f"Evaluation metric #{i}: {metric}")

            if not get_tf_flag(i):
                self._refresh_teacher_qa()
                single_result["history"]["teacher_history"] = self._teacher_qa_list

            metrics.append(metric)
            single_results.append(single_result)

            example_ctx = f"Example #{i + 1}: \n" + model.rebuild_context(self._teacher_qa_list)
            instruction_w_examples += example_ctx

        return self._pack_results(metrics, single_results, teacher_forcing_mode)

    # TODO: Deprecated
    def test_multi_time(self, model, times, teacher_forcing_mode="l0"):
        # teacher forcing options:
        # "l0": no teacher forcing, context is cleared after each test
        # "l1": naive teacher forcing, context is cleared after each test
        # "l2": no teacher forcing during the current test, previous context is used as
        #       initial prompt after forced
        # "l3": similar to "l4" but the final test runs in the "l2" mode
        # "l4": full teacher forcing, previous context is used as initial prompt after forced

        assert teacher_forcing_mode in ["l0", "l1", "l2", "l3", "l4"], teacher_forcing_mode

        if teacher_forcing_mode in ["l0", "l1"]:
            return self._independent_test(model, times, teacher_forcing_mode)

        return self._context_kept_test(model, times, teacher_forcing_mode)

    def _add_examples(self, instruction, qa_lists, rebuild_context_func):
        if not qa_lists:
            return instruction

        instruction_w_examples = instruction \
            + "\nHere are some examples (the right answer for each example is different):\n"

        for i, qa_list in enumerate(qa_lists):
            example_ctx = f"Example #{i + 1}: \n" + rebuild_context_func(qa_list)
            instruction_w_examples += example_ctx

        return instruction_w_examples

    def _init_teacher_qa_lists(self, num_examples):
        if not num_examples:
            return []

        teacher_qa_lists = []
        for example_case in self.test_cases[- num_examples:]:
            self.reset(example_case)
            self._refresh_teacher_qa()
            teacher_qa_lists.append(self._teacher_qa_list)

        return teacher_qa_lists

    def _if_ckpt_exist(self):
        return osp.exists(osp.join(self.output_dir, "last_checkpoint"))

    def _save_ckpt(self, ckpt, filename):
        logger.info("Saving ckpt to {}".format(osp.join(self.output_dir, filename)))

        pkl.dump(ckpt, open(osp.join(self.output_dir, filename), mode="wb"))
        with open(osp.join(self.output_dir, "last_checkpoint"), mode="w") as f:
            f.write(filename)

    def _load_ckpt(self):
        with open(osp.join(self.output_dir, "last_checkpoint"), mode="r") as f:
            filename = f.readline()

        logger.info("Loading ckpt from {}".format(osp.join(self.output_dir, filename)))

        ckpt = pkl.load(open(osp.join(self.output_dir, filename), mode="rb"))
        # TODO: if the evaluation is finished, there will be no need for `teacher_qa_lists`
        if filename == "results_final.pkl":
            return ckpt, []

        return ckpt["single_results"], ckpt["teacher_qa_lists"]

    def test_with_examples(
        self, model, times, num_examples=0, teacher_forcing=False, weak_tg_chances=0, resume=False
    ):
        assert times <= len(self.test_cases), self.test_cases
        assert num_examples <= len(self.test_cases), self.test_cases

        if resume and self._if_ckpt_exist():
            single_results, teacher_qa_lists = self._load_ckpt()
            logger.info(f"Resume at #{len(single_results)}")
        else:
            single_results = []
            teacher_qa_lists = self._init_teacher_qa_lists(num_examples)

        start = len(single_results)

        for i, test_case in enumerate(self.test_cases[start: times]):
            i += start + 1

            self.pre_each_test(
                model,
                instruction=self.default_instruction,
                test_case=test_case,
                example_qa_lists=teacher_qa_lists
            )

            metric, single_result = self.naive_test(
                model, teacher_forcing, weak_tg_chances, self.default_instruction
            )
            logger.info(f"Evaluation metric #{i}: {metric}")

            if num_examples:
                teacher_qa_lists = teacher_qa_lists[1:]
                if not teacher_forcing:
                    self._refresh_teacher_qa()
                    # single_result["history"]["teacher_history"] = deepcopy(self._teacher_qa_list)
                teacher_qa_lists.append(self._teacher_qa_list)

            single_results.append(single_result)

            if self.save_period > 0 and not i % self.save_period:
                self._save_ckpt(
                    {"single_results": single_results, "teacher_qa_lists": teacher_qa_lists},
                    f"ckpt_{i}.pkl"
                )

        metric, full_result = self._pack_results(
            single_results, teacher_forcing_mode=teacher_forcing
        )

        if self.save_period >= 0:
            self._save_ckpt(full_result, "results_final.pkl")

            logger.info("Saving json to {}".format(osp.join(self.output_dir, "results_final.json")))
            json.dump(
                full_result,
                open(osp.join(self.output_dir, "results_final.json"), mode="w"),
                cls=InvalidEncoder
            )

        logger.info(f"Final metrics: {metric}")

        return metric, full_result

    @property
    @abc.abstractmethod
    def default_instruction(self):
        pass

    @abc.abstractmethod
    def _get_prompt(self):
        pass

    @abc.abstractmethod
    def _extract_answer(self):
        """
            return extracted answer with the correct format after checking for
            errors such as invalid format or value.
            return `Invalid` accordingly if erros are found.
        """
        pass

    @abc.abstractmethod
    def calc_metric_tf(self, answer_list, target_list):
        pass

    @abc.abstractmethod
    def calc_metric_no_tf(self, answer_list, target_list):
        pass

    @abc.abstractmethod
    def _test_no_tf(self, model, weak_tg_chances=0):
        pass

    @abc.abstractmethod
    def _test_tf(self, model):
        pass

    @abc.abstractmethod
    def naive_test(self, model, teacher_forcing=False, weak_tg_chances=0, instruction=None):
        pass
