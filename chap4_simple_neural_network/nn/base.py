class Node:
    def __init__(self):
        pass


class DataNode(Node):
    """Parent class for Parameter and Constant nodes"""

    def __init__(self, data):
        self.parents = []
        self.data = data

    def _forward(self, *inputs):
        return self.data

    @staticmethod
    def _backward(gradients, *inputs):
        return []

    def item(
        self,
    ):
        assert self.data.size == 1, f"Cannot convert shape {self.data.shape} to a scalar"
        return self.data.item()


class FunctionNode(Node):
    """Parent class for nodes whose values depend on other nodes"""

    def __init__(self, *parents):
        assert all(isinstance(parent, Node) for parent in parents)
        self.parents = parents
        self.data = self._forward(*(parent.data for parent in parents))

    @staticmethod
    def _forward(
        *inputs,
    ):
        raise NotImplementedError

    @staticmethod
    def _backward(gradient, *inputs):
        raise NotImplementedError