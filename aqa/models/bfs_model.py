from .dfs_model import extract_int


class BFSModel():
    def __init__(self):
        self.reset("")

    def _get_adj_nodes(self, prompt):
        return extract_int(prompt.split(".")[0])

    def reset(self, instruction=""):
        self.history = [0]
        self.node_queue = []  # dfs trajectory

    def __call__(self, prompt):
        adj_nodes = self._get_adj_nodes(prompt)
        # unvisited_adj_nodes = [node for node in adj_nodes if node not in set(self.history)]

        for node in adj_nodes:
            if node not in set(self.history):
                self.node_queue.append(node)

        while self.node_queue[0] in set(self.history):
            self.node_queue.pop(0)

        # if no node exist, end exploring
        # TODO: use assert
        if len(self.node_queue) == 0:
            return "null"

        next_node = self.node_queue.pop(0)

        self.history.append(next_node)
        return str(next_node)

    def force(self, new_reply):
        self.node_queue.insert(0, self.history[-1])
        self.node_queue.pop(self.node_queue.index(int(new_reply)))
        self.history[-1] = int(new_reply)
