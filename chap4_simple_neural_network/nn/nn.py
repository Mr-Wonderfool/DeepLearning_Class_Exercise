import numpy as np
from base import FunctionNode, DataNode, Node


class Constant(DataNode):
    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        super().__init__(data)


class Parameter(DataNode):
    def __init__(self, *shape):
        # employ Xavier initialization
        limit = np.sqrt(3.0 / np.mean(shape))
        data = np.random.uniform(low=-limit, high=limit, size=shape)
        super().__init__(data)

    def update(self, direction, multiplier):
        assert isinstance(direction, Constant)
        self.data += multiplier * direction.data

class AddBias(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2
        return inputs[0] + inputs[1]
    @staticmethod
    def _backward(gradient, *inputs):
        # ! shape for bias is automatically broadcasted across batch
        assert gradient.shape == inputs[0].shape
        return [gradient, np.sum(gradient, axis=0, keepdims=True)]

class Linear(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2
        return np.dot(inputs[0], inputs[1])

    @staticmethod
    def _backward(gradient, *inputs):
        # backprop with matmul, x: (b, f), W: (f, f'), gradient: (b, f')
        assert gradient.shape[0] == inputs[0].shape[0]
        assert gradient.shape[1] == inputs[1].shape[1]
        # grad_x, grad_w
        return [np.dot(gradient, inputs[1].T), np.dot(inputs[0].T, gradient)]


class ReLU(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 1
        return np.maximum(inputs[0], 0)

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient * np.where(inputs[0] > 0, 1.0, 0.0)]


class SquareLoss(FunctionNode):
    """Mean square loss for two matrices with shape (b, f)"""

    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2
        return 0.5 * np.mean((inputs[0] - inputs[1]) ** 2)

    @staticmethod
    def _backward(gradient, *inputs):
        # gradient must be a scalar
        assert np.asarray(gradient).ndim == 0
        N = inputs[0].size
        return [gradient * (inputs[0] - inputs[1]) / N, gradient * (inputs[1] - inputs[0]) / N]

def backprop(loss: Node, parameters: list[Node]):
    """ Backprop in computational graph """
    assert isinstance(loss, SquareLoss), f"Currently only support square loss"
    # track usage of node
    assert not hasattr(loss, "used"), f"Node has already been used in backprop"
    loss.used = True
    nodes: set[DataNode] = set()
    tape: list[DataNode] = []
    
    def visit(node: FunctionNode):
        # depth first search
        if node not in nodes:
            for parent in node.parents:
                visit(parent)
            nodes.add(node)
            tape.append(node)
    # DFS to traverse graph
    visit(loss)
    nodes.update(set(parameters))
    
    grads = {node: np.zeros_like(node.data) for node in nodes}
    grads[loss] = 1.0
    
    # traverse from leaf to root
    for node in reversed(tape):
        parent_grads = node._backward(
            grads[node], *(parent.data for parent in node.parents)
        )
        for parent, parent_grad in zip(node.parents, parent_grads):
            grads[parent] += parent_grad

    return [Constant(grads[parameter]) for parameter in parameters]