from copy import deepcopy

from aqa.models import DFSModel
from aqa.utils import Invalid

from .build import BENCHMARKS
from .traverse import TraverseGraph


@BENCHMARKS.register()
class DFS(TraverseGraph):
    def __init__(
        self, node_num=4, explain_algo=True, mcq=False, provide_state=False,
        format_tolerant=True, max_retry=0, max_step=None,
        verbose=True, output_dir=None, save_period=-1
    ):
        super(DFS, self).__init__(
            node_num, explain_algo, mcq, provide_state,
            format_tolerant, max_retry, max_step,
            verbose, output_dir, save_period
        )
        self.teacher = DFSModel()

    @property
    def default_instruction(self):
        instruction = "You are required to visit all the nodes in an undirected non-cyclic graph." \
                      "An undirected non-cyclic garph contains a set of node, and a set of edges that each connects a pair of nodes. " \
                      "All edges are undirected, so that you can move from one node to the other connected by the edge in either direction. " \
                      "Every time you visit a node, you will be given the adjacent nodes connected to this node. " \
                      "You can only reply with a integer number indicating which node to be visited next. " \
                      "Don't explain your answer. " \
                      "Try traverse the entire graph in as few rounds as possible." \
                      "You are currently on the node 0." \

        if self.explain_algo:
            instruction += "\nYou should use depth first search algorithm, each time you should " \
                           "select a node you have not moved to. If all nodes adjacent to the " \
                           "current node have been visited, you should back track to the node " \
                           "through which you entered this node for the first time. "

        return instruction

    def _get_valid_nodes(self, next_node, visited_nodes):
        return set(self._get_adj_nodes(next_node))

    def _get_prompt_when_invalid(self, valid_nodes):
        prompt = "Invalid reply. Try again. You can only reply with a " \
                 "integer number indicting the node adjacent to the current node."
        if self.mcq:
            valid_nodes = [str(node) for node in valid_nodes]
            prompt += " Valid nodes: {}.".format(", ".join(valid_nodes))

        return prompt

    def _get_prompt_when_weak_tg(self, valid_nodes):
        prompt = "Your output is not optimal. You should follow depth-first-search algorithm."
        prompt += " Try again. You can only reply with a " \
                  "integer number indicting the node adjacent to the current node."
        if self.mcq:
            valid_nodes = [str(node) for node in valid_nodes]
            prompt += " Valid nodes: {}.".format(", ".join(valid_nodes))

        return prompt

    def _init_stack(self):
        return [self._start_node]

    def _update_stack(self, next_node, stack, visited_nodes):
        # doesn't change `stack` in-place
        assert isinstance(stack, list) and len(stack), stack
        stack = deepcopy(stack)
        # backtrace
        if len(stack) > 1 and next_node == stack[-2]:
            return stack[:-1]

        stack.append(next_node)
        return stack

    def _check_dfs(self, next_node, stack, visited_nodes):
        '''
        Check whether `next_node` follows DFS
        Will assume the previous steps in `node_history` already follow DFS

        Return
        - boolean: if selected interface follows dfs
        '''
        if isinstance(next_node, Invalid):
            return False

        curr_node = stack[-1]
        adj_nodes = self._get_adj_nodes(curr_node)

        # check if model selected node following dfs path
        # i.e. select unvisited child node or parent node
        unvisited_adj_nodes = set(adj_nodes).difference(set(visited_nodes))
        if len(unvisited_adj_nodes):
            # should visit child node
            return next_node in unvisited_adj_nodes

        # if all child have been fisited,
        # check if model is visiting its parent node in the history stack

        # `curr_node` should be the root only when there are children of the root unvisited
        assert curr_node != self._start_node
        # should visit father node
        if_dfs = (next_node == stack[-2])

        return if_dfs

    _init_stack_or_queue = _init_stack
    _update_stack_or_queue = _update_stack
    _check_algo = _check_dfs
