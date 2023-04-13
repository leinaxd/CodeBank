from typing import List


class FixedGraph:
    def __init__(self, adjacencyMatrix):
        self.adjacencyMatrix = adjacencyMatrix

class GrowingGraph:
    """
    A list-implemented graph, whose nodes can be updated
    """
    def __init__(self):
        self.nodes = {}
    def addNode(self, symbol):
        assert symbol not in self.nodes, f"symbol {symbol} already implemented"
        pass
class Node:
    """
    A node is a holder for a function, a weight or just empty
    """
    def __init__(self, id): self.id = id
    def __str__(self): return str(self.id)

class BaseGraph(object):
    """
    A weighted Graph is a graph with a weight for each arc.
    """
    def __init__(self, nodes:List[Node]):
        assert all([isinstance(n, Node) for n in nodes])
        self.nodes = nodes
    def addNodes(self, symbols):
        for symbol in symbols:
            assert symbol not in self.nodes, f"symbol {symbol} already implemented"
        self.nodes[symbol]

    def __str__(self):
        nodes  = f"Node number: Symbol\n"
        nodes += "\n".join([f"{i}: {node}" for i, node in enumerate(self.nodes)])
        return nodes
    

if __name__=='__main__':
    nodes = [Node(name) for name in 'abcdef']

    graph = BaseGraph(nodes)

    print(graph)