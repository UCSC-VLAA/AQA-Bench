from copy import deepcopy

from aqa.models import DFSModel
from aqa.utils import Invalid

from .build import BENCHMARKS
from .dfs import DFS


@BENCHMARKS.register()
class CaveDFS(DFS):
    def __init__(
        self, node_num=4, explain_algo=False, mcq=False, provide_state=False,
        format_tolerant=True, max_retry=0, max_step=None,
        verbose=True, output_dir=None, save_period=-1
    ):
        super(CaveDFS, self).__init__(
            node_num, explain_algo, mcq, provide_state,
            format_tolerant, max_retry, max_step,
            verbose, output_dir, save_period
        )
        assert not self.explain_algo

    @property
    def default_instruction(self):
        instruction = "There is an expansive underground cave system in which each cave is uniquely numbered and interconnected by tunnels. " \
                      "Every time you visit a cave, you will know the adjacent caves directly connected to this one. " \
                      "You can only reply with a integer number indicating which cave to be visited next. " \
                      "Don't explain your answer. " \
                      "Your objective is to explore every cave, starting from cave 0. " \
                      "Try to visit all the caves in as few rounds as possible." \
                      "You are currently in the cave 0."

        return instruction

    def _get_prompt(self, next_node, visited_nodes):
        '''
        Generate prompt used in exploration step

        Return: prompt (string)
        '''

        if len(set(visited_nodes + [next_node])) == len(self._graph.nodes):
            return "Well Done. You have visited all the caves. " \
                   "Total number of steps: {}".format(len(visited_nodes[1:] + [next_node]))

        adj_nodes = self._get_adj_nodes(next_node)

        prompt = "Adjacent caves: {}.".format(", ".join([str(i) for i in adj_nodes]))

        if self.provide_state:
            unvisited_adj_nodes = set(adj_nodes).difference(set(visited_nodes))
            if len(unvisited_adj_nodes) == 0:
                prompt += " You have visited all caves adjacent to this cave."
            else:
                prompt += " You have not visited cave {}." \
                          .format(", ".join([str(i) for i in unvisited_adj_nodes]))
        if self.mcq:
            valid_nodes = self._get_valid_nodes(next_node, visited_nodes)
            valid_nodes = [str(node) for node in valid_nodes]

            prompt += " Valid caves: {}.".format(", ".join(valid_nodes))

        return prompt

    def _get_prompt_when_invalid(self, valid_nodes):
        prompt = "Invalid reply. Try again. You can only reply with a " \
                 "integer number indicting the cave adjacent to the current cave."
        if self.mcq:
            valid_nodes = [str(node) for node in valid_nodes]
            prompt += " Valid caves: {}.".format(", ".join(valid_nodes))

        return prompt

    def _get_prompt_when_weak_tg(self, valid_nodes):
        prompt = "Your output is not optimal. You should follow depth-first-search algorithm."
        prompt += " Try again. You can only reply with a " \
                  "integer number indicting the cave adjacent to the current cave."
        if self.mcq:
            valid_nodes = [str(node) for node in valid_nodes]
            prompt += " Valid caves: {}.".format(", ".join(valid_nodes))

        return prompt
