from copy import deepcopy

from aqa.models import BFSModel
from aqa.utils import Invalid

from .build import BENCHMARKS
from .traverse import TraverseGraph


@BENCHMARKS.register()
class BFS(TraverseGraph):
    def __init__(
        self, node_num=4, explain_algo=True, mcq=False, provide_state=False,
        format_tolerant=True, max_retry=0, max_step=None,
        verbose=True, output_dir=None, save_period=-1
    ):
        super(BFS, self).__init__(
            node_num, explain_algo, mcq, provide_state,
            format_tolerant, max_retry, max_step,
            verbose, output_dir, save_period
        )
        self.teacher = BFSModel()

    @property
    def default_instruction(self):
        instruction = "You are required to visit all the nodes in an undirected non-cyclic graph." \
                      "An undirected non-cyclic garph contains a set of node, and a set of edges that each connects a pair of nodes. " \
                      "Every time you visit a node, you will be given the adjacent nodes connected to this node. " \
                      "You can only visit nodes that are adjacent to the already visited nodes. " \
                      "You can only reply with a integer number indicating which node to be visited next. " \
                      "Don't explain your answer. " \
                      "Try traverse the entire graph in as few rounds as possible." \
                      "You are currently on the node 0." \

        if self.explain_algo:
            instruction += "\nYou should use breadth first search algorithm. " \
                           "The algorithm works as follows:\n" \
                           "1. Initialize a queue data structure and add the starting node to the queue.\n" \
                           "2. While the queue is not empty, visit the first node and remove it from the queue.\n" \
                           "3. For nodes adjacent to the removed vertex, add the unvisited ones to the queue.\n" \
                           "4. Repeat steps 2-3 until the queue is empty."

        return instruction

    def _get_valid_nodes(self, next_node, visited_nodes):
        valid_nodes = set(
            sum(
                [(self._get_adj_nodes(node) + [node]) for node in visited_nodes + [next_node]],
                start=[]
            )
        )
        assert self._start_node in valid_nodes

        return valid_nodes

    def _get_prompt_when_invalid(self, valid_nodes):
        prompt = "Invalid reply. Try again. You can only reply with a " \
                 "integer number indicting the node adjacent to the visited node."
        if self.mcq:
            valid_nodes = [str(node) for node in valid_nodes]
            prompt += " Valid nodes: {}.".format(", ".join(valid_nodes))

        return prompt

    def _get_prompt_when_weak_tg(self, valid_nodes):
        prompt = "Your output is not optimal. You should follow breadth-first-search algorithm."
        prompt += " Try again. You can only reply with a " \
                  "integer number indicting the node adjacent to the visited node."
        if self.mcq:
            valid_nodes = [str(node) for node in valid_nodes]
            prompt += " Valid nodes: {}.".format(", ".join(valid_nodes))

        return prompt

    def _init_queues(self):
        return self._get_adj_nodes(self._start_node), []

    def _update_queues(self, next_node, old_new_queues, visited_nodes):
        # doesn't change queues in-place
        assert isinstance(old_new_queues, tuple) and len(old_new_queues) == 2, old_new_queues
        old_queue, new_queue = deepcopy(old_new_queues)
        assert next_node in old_queue
        old_queue.pop(old_queue.index(next_node))
        assert next_node not in old_queue

        new_queue += [
            node
            for node in self._get_adj_nodes(next_node)
            if node not in (visited_nodes + new_queue)
        ]

        if not old_queue:
            old_queue = new_queue
            new_queue = []

        return old_queue, new_queue

    def _check_bfs(self, next_node, old_new_queues, visited_nodes):
        '''
        Check whether `next_node` follows BFS
        Will assume the previous steps in `node_history` already follow BFS

        Return
        - boolean: if selected interface follows bfs
        '''

        assert isinstance(old_new_queues, tuple) and len(old_new_queues) == 2, old_new_queues

        if isinstance(next_node, Invalid):
            return False

        old_queue, _ = old_new_queues
        assert old_queue, old_queue

        return next_node in old_queue

    _init_stack_or_queue = _init_queues
    _update_stack_or_queue = _update_queues
    _check_algo = _check_bfs
