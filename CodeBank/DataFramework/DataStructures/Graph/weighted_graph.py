class Edge:
    """
    A weighted edge
    """
    def __init__(self, node_a:int, node_b:int, weight=None):
        assert isinstance(node_a, int) and isinstance(node_b, int)
        self.node_a = node_a
        self.node_b = node_b
        self.weight = weight
    def __str__(self): return f"{self.node_a}->{self.node_b}|{self.weight}"
    def __repr__(self): return f"{self.node_a}->{self.node_b}|{self.weight}"


class WeightedGraph(object):
    """
    A weighted Graph is a graph with a weight for each arc.
    """
    def __init__(self):
        self.edges = []

    def __call__(self, edge:Edge):
        """Adds an edge to the graph"""
        assert isinstance(edge, Edge)
        self.edges.append(edge)

    def __str__(self):
        nodes  = f"Edge ID\t:   δ   | weight\n"
        nodes += "\n".join([f"{i}\t: {edge}\t" for i, edge in enumerate(self.edges)])
        return nodes


if __name__=='__main__':
    graph = WeightedGraph()
    graph(Edge(0, 1, 2))
    graph(Edge(0, 3, 5))
    graph(Edge(3, 1, 1))
    graph(Edge(3, 6, 5))
    graph(Edge(3, 4, 6))
    graph(Edge(1, 6, 2))
    graph(Edge(4, 2, 4))
    graph(Edge(4, 5, 4))
    graph(Edge(2, 1, 3))
    graph(Edge(5, 2, 6))
    graph(Edge(5, 6, 3))
    graph(Edge(6, 4, 7))
    
    print(graph)