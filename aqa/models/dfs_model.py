import random
import re


def extract_int(s):
    def isint(word):
        try:
            int(word)
            return True
        except ValueError:
            return False
    return [int(word) for word in re.split("\n|,| |\t|\(|\)|\[|\]", s) if isint(word)]


class DFSModel():
    def __init__(self):
        self.reset("")

    def _get_adj_nodes(self, prompt):
        return extract_int(prompt.split(".")[0])

    def reset(self, instruction=""):
        self.history = [0]
        self.node_stack = [0]  # dfs trajectory

    def __call__(self, prompt):
        adj_nodes = self._get_adj_nodes(prompt)
        unvisited_adj_nodes = [node for node in adj_nodes if node not in set(self.history)]

        if len(unvisited_adj_nodes):
            next_node = unvisited_adj_nodes[0]
            self.history.append(next_node)
            self.node_stack.append(next_node)

            return str(next_node)

        # backtrack to parent node
        # if no node exist, end exploring
        if len(self.node_stack) == 0:
            return "null"

        self.node_stack.pop()
        self.history.append(self.node_stack[-1])
        return str(self.node_stack[-1])

    def force(self, new_reply):
        self.history[-1] = int(new_reply)
        self.node_stack[-1] = int(new_reply)
