import networkx
import re

from loguru import logger

from aqa.utils import Invalid, FormatInvalid, ValueInvalid

from .benchmark import Benchmark


# TODO: refine or just remove provide_state
class TraverseGraph(Benchmark):
    def __init__(
        self, node_num=4, explain_algo=True, mcq=False, provide_state=False,
        format_tolerant=True, max_retry=0, max_step=None,
        verbose=True, output_dir=None, save_period=-1
    ):
        super(TraverseGraph, self).__init__(
            format_tolerant, max_retry, max_step, verbose, output_dir, save_period
        )
        self.node_num = node_num
        self.explain_algo = explain_algo
        self.mcq = mcq
        self.provide_state = provide_state

        # Set `self.teacher` here

    def reset(self, test_case=None):
        super(TraverseGraph, self).reset(test_case)

        # TODO: maybe choose random node as the starting node
        self._start_node = 0

        if test_case is None:
            logger.info("Generating random graph.")
            self._graph = networkx.random_tree(self.node_num).to_undirected()
        else:
            logger.info("Using pre-generated random graph.")
            self._graph = networkx.Graph()
            assert len(test_case["nodes"]) == self.node_num, test_case["nodes"]
            self._graph.add_nodes_from(test_case["nodes"])
            self._graph.add_edges_from(test_case["edges"])

    def _get_adj_nodes(self, curr_node):
        return [n for _, n in self._graph.edges(curr_node)]

    def _get_valid_nodes(self, next_node, visited_nodes):
        raise NotImplementedError

    def _get_prompt(self, next_node, visited_nodes):
        '''
        Generate prompt used in exploration step

        Return: prompt (string)
        '''

        if len(set(visited_nodes + [next_node])) == len(self._graph.nodes):
            return "Well Done. You have visited all the nodes in the graph. " \
                   "Total number of steps: {}".format(len(visited_nodes[1:] + [next_node]))

        adj_nodes = self._get_adj_nodes(next_node)

        prompt = "Adjacent nodes: {}.".format(", ".join([str(i) for i in adj_nodes]))

        if self.provide_state:
            unvisited_adj_nodes = set(adj_nodes).difference(set(visited_nodes))
            if len(unvisited_adj_nodes) == 0:
                prompt += " You have visited all nodes adjacent to this node."
            else:
                prompt += " You have not visited node {}." \
                          .format(", ".join([str(i) for i in unvisited_adj_nodes]))
        if self.mcq:
            valid_nodes = self._get_valid_nodes(next_node, visited_nodes)
            valid_nodes = [str(node) for node in valid_nodes]

            prompt += " Valid nodes: {}.".format(", ".join(valid_nodes))

        return prompt

    def _get_prompt_when_invalid(self, valid_nodes):
        raise NotImplementedError

    def _get_prompt_when_weak_tg(self, valid_nodes):
        raise NotImplementedError

    def _refresh_teacher_qa(self):
        super(TraverseGraph, self)._refresh_teacher_qa()

        response = ""
        prompt = self._get_prompt(self._start_node, [])
        decov_sum = 0.0  # ignore the starting node

        next_node = self._start_node
        node_history = [self._start_node]

        # while exist node not visited
        while len(set(self._graph.nodes).difference(set(node_history))) != 0:
            response = self.teacher(prompt)
            next_node = int(response)

            self._teacher_qa_list.append((prompt, next_node))
            prompt = self._get_prompt(next_node, node_history)

            decov_sum += self._calc_decoverage(node_history + [next_node])
            node_history.append(next_node)

        self._teacher_qa_list.append((prompt, None))

        # remove start node in node_history
        node_history = node_history[1:]

        return decov_sum

    def _extract_answer(self, reply, valid_nodes):
        # parse reply from model and return the formatted answer
        # return an `Invalid` if failed to do so
        if self.format_tolerant:
            nums = re.findall(r'\d+', reply)
            if not len(nums):
                return FormatInvalid(reply)

            next_node = int(nums[0])

            if next_node not in valid_nodes:
                return ValueInvalid(next_node)
            return next_node

        try:
            next_node = int(reply)

            if next_node not in valid_nodes:
                return ValueInvalid(next_node)
            return next_node
        except ValueError:
            return FormatInvalid(reply)

    def _init_stack_or_queue(self):
        raise NotImplementedError

    def _update_stack_or_queue(self, next_node, stack_or_queue, node_history):
        raise NotImplementedError

    def _check_algo(self, next_node, stack_or_queue, visited_nodes):
        '''
        Check whether `next_node` follows the traversing algorithm (BFS/DFS)
        Will assume the previous steps in `node_history` already follow the traversing algorithm

        Return
        - boolean: if selected interface follows dfs
        '''
        raise NotImplementedError

    def _calc_decoverage(self, visited_nodes):
        assert self._start_node in visited_nodes
        return 1 - len(set(visited_nodes)) / len(self._graph.nodes)

    def calc_metric_no_tf(self, node_history):
        assert len(node_history) > 0

        decov_list = [self._calc_decoverage([self._start_node])]
        highest_cnt = 0
        is_following_algo = True

        stack_or_queue = self._init_stack_or_queue()

        for idx, node in enumerate(node_history):
            if isinstance(node, Invalid):
                assert idx == len(node_history) - 1, \
                    f"Only the last node can be Invalid without teacher forcing. {node_history}"
                break

            # `is_following_algo` will remain `True` until `model` stops following bfs
            if is_following_algo:
                is_following_algo = self._check_algo(
                    node, stack_or_queue, [self._start_node] + node_history[:idx]
                )
            if is_following_algo:
                highest_cnt = idx + 1
                stack_or_queue = self._update_stack_or_queue(
                    node, stack_or_queue, [self._start_node] + node_history[:idx]
                )

            decov = self._calc_decoverage([self._start_node] + node_history[:idx + 1])
            assert decov <= decov_list[-1], "`decov_list` should be a non-ascent sequence"
            decov_list.append(decov)

        # ignore the starting node
        # dont ignore last invalid
        acc = highest_cnt / len(node_history)
        min_decov = decov_list[-1]
        # ignore the starting node
        decov_list = decov_list[1:]
        sum_decov = sum(decov_list)

        metric = {
            "acc": acc,
            "min_decov": min_decov,
            "sum_decov": sum_decov,
            # "decov_list": decov_list
        }
        return metric

    def calc_metric_tf(self, node_history, teacher_node_history):
        assert len(node_history) > 0

        decov_list = [self._calc_decoverage([self._start_node])]
        cnt = 0

        stack_or_queue = self._init_stack_or_queue()

        for idx, (node, teacher_node) in enumerate(
            zip(node_history, teacher_node_history)
        ):
            is_following_algo = self._check_algo(
                node, stack_or_queue, [self._start_node] + teacher_node_history[:idx]
            )

            if is_following_algo:
                cnt += 1

            stack_or_queue = self._update_stack_or_queue(
                teacher_node, stack_or_queue, [self._start_node] + teacher_node_history[:idx]
            )

            if isinstance(node, Invalid):
                decov = decov_list[-1]
            else:
                decov = self._calc_decoverage(
                    [self._start_node] + teacher_node_history[:idx] + [node]
                )
            assert decov <= decov_list[-1], "`decov_list` should be a non-ascent sequence"
            decov_list.append(decov)

        # ignore the starting node
        # dont ignore last invalid
        acc = cnt / len(node_history)
        min_decov = decov_list[-1]
        # ignore the starting node
        decov_list = decov_list[1:]
        sum_decov = sum(decov_list)

        metric = {
            "acc": acc,
            "min_decov": min_decov,
            "sum_decov": sum_decov,
            # "decov_list": decov_list
        }
        return metric

    def _test_no_tf(self, model, weak_tg_chances=0):
        '''
        Return:
        - accuracy: percentage of node selected following the traversing algorithm (BFS/DFS)
        - decov_list: list of (1 - coverages)
        - trace of node explored by model
        '''
        prompt = self._get_prompt(self._start_node, [])
        node_history = []

        retry_cnt = 0
        # for weak tg
        weak_tg_cnt = 0
        do_algo_check = weak_tg_chances > 0
        stack_or_queue = self._init_stack_or_queue()

        valid_nodes = self._get_valid_nodes(self._start_node, [])

        while (
            len(set([self._start_node] + node_history)) != len(self._graph.nodes)
            and (self.max_step is None or len(node_history) < self.max_step)
            and retry_cnt < (self.max_retry + 1)
        ):
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)
            self.dialog_logger.info(A=reply)

            # start processing response in this iteration
            next_node = self._extract_answer(reply, valid_nodes)

            # if `reply` is formatted in `_extract_answer`, force the new reply
            if not isinstance(next_node, FormatInvalid) \
               and str(getattr(next_node, "output", next_node)) != reply:
                assert self.format_tolerant
                formatted = str(getattr(next_node, "output", next_node))
                logger.info(f"Format tolerance enabled, force the model reply to {formatted}.")
                model.force(formatted)

            if isinstance(next_node, Invalid):
                prompt = self._get_prompt_when_invalid(valid_nodes)
                retry_cnt += 1
                continue

            if do_algo_check:
                if weak_tg_cnt < weak_tg_chances:
                    is_following_algo = self._check_algo(
                        next_node, stack_or_queue, [self._start_node] + node_history
                    )
                    if not is_following_algo:
                        weak_tg_cnt += 1
                        prompt = self._get_prompt_when_weak_tg(valid_nodes)
                        continue

                weak_tg_cnt = 0

                do_algo_check = is_following_algo

                if do_algo_check:
                    stack_or_queue = self._update_stack_or_queue(
                        next_node, stack_or_queue, [self._start_node] + node_history
                    )

            valid_nodes = self._get_valid_nodes(next_node, [self._start_node] + node_history)
            prompt = self._get_prompt(next_node, [self._start_node] + node_history)
            node_history.append(next_node)
            retry_cnt = 0

        self.dialog_logger.info(Q=prompt)

        if isinstance(next_node, Invalid):
            node_history.append(next_node)  # save the last invalid
            logger.info("Max retry times reached, stop interaction now.")
        elif len(set([self._start_node] + node_history)) != len(self._graph.nodes):
            # target not achieved
            logger.info("Max steps reached, stop the interaction now.")

        return node_history

    def _test_tf(self, model):
        valid_nodes = self._get_valid_nodes(self._start_node, [])
        node_history = []
        teacher_node_history = []

        optim_decov_sum = self._refresh_teacher_qa()

        # no retry when teacher forcing
        for prompt, teacher_reply in self._teacher_qa_list[:-1]:
            self.dialog_logger.info(Q=prompt)

            reply = model(prompt)

            model.force(str(teacher_reply))
            self.dialog_logger.info(A=reply, T=teacher_reply)

            next_node = self._extract_answer(reply, valid_nodes)

            node_history.append(next_node)
            teacher_node_history.append(teacher_reply)
            valid_nodes = self._get_valid_nodes(
                teacher_reply, [self._start_node] + teacher_node_history
            )

        self.dialog_logger.info(Q=self._teacher_qa_list[-1][0])

        return node_history, teacher_node_history, optim_decov_sum

    def naive_test(self, model, teacher_forcing=False, weak_tg_chances=0, instruction=None):
        logger.info("Nodes: {}, Edges: {}".format(self._graph.nodes, self._graph.edges))

        if teacher_forcing:
            model_node_history, teacher_node_history, optim_decov_sum = self._test_tf(model)
            metric = self.calc_metric_tf(model_node_history, teacher_node_history)
        else:
            teacher_node_history = []
            model_node_history = self._test_no_tf(model, weak_tg_chances)
            metric = self.calc_metric_no_tf(model_node_history)

        result = self._get_result(
            metric, model_node_history, teacher_node_history,
            model.history, teacher_forcing, instruction
        )

        return metric, result

    # TODO: maybe `teacher_answer_list`, `_teacher_qa_list` should be None if no `teacher_forcing`
    def _get_result(
        self, metric, answer_list, teacher_answer_list,
        model_history, teacher_forcing, instruction=None
    ):
        result = super(TraverseGraph, self)._get_result(
            metric, answer_list, teacher_answer_list, model_history, teacher_forcing, instruction
        )

        result["env"].update(dict(
            # optim_decov_sum=optim_decov_sum if teacher_forcing else None,
            nodes=list(self._graph.nodes),
            edges=list(self._graph.edges),
            start_node=self._start_node,
            mcq=self.mcq,
            explain_algo=self.explain_algo,
            provide_state=self.provide_state,
        ))

        return result

    def _pack_results(self, single_results, teacher_forcing_mode):
        metrics, full_result = super(TraverseGraph, self)._pack_results(
            single_results, teacher_forcing_mode
        )

        full_result["env"].update(dict(
            nodes=list(self._graph.nodes),
            edges=list(self._graph.edges),
            start_node=self._start_node,
        ))

        return metrics, full_result
