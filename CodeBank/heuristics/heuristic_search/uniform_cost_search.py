from typing import List, Union, Dict
from CodeBank.DataFramework.DataStructures.Graph import WeightedGraph, Edge

class SearchGraph(WeightedGraph):
    def __init__(self, heuristic):
        super().__init__()
        self.heuristic = heuristic
    
    def search(self, start_node:int, goal_nodes:Union[int, List[int]]):
        if isinstance(goal_nodes, int): goal_nodes = [goal_nodes]
        return self.heuristic(self, start_node, goal_nodes)

class UniformCostSearch:
    """Uniform Cost Search Algorithm
    Features:
        - Uninformed search
          (A* heuristic or informed)
        - Optimal goal
    
    Requirements:
        - All edge weights (costs) must be POSITIVE

    Traverse a graph Searching the optimal goal node 

    Sources:
        https://www.geeksforgeeks.org/uniform-cost-search-dijkstra-for-large-graphs/
        https://www.youtube.com/watch?v=dRMvK76xQJI
    Complexity:
        O(m^(1+floor(l/e)))
        where
        m: max number of neightbors a node has
        l: length of the shortest path to the goal state
        e: is the least cost of an edge
    """
    def __init__(self, returnPath=True, verbose=False):
        self.verbose = verbose
        self.returnPath = returnPath


    def neighbors(self, edges: List[Edge]) -> Dict[int, List[Edge]]:
        """returns a list of neightbors for each node_a"""
        nodes = {}
        for edge in edges:
            try:
                nodes[edge.node_a].append( edge ) #append a neighbor of a
            except:
                nodes[edge.node_a] = [edge]
        return nodes
    def __call__(self, graph:SearchGraph, start:int, goal:List[int]):
        assert start not in goal, 'Start and goal cannot be the same'
        if self.returnPath: return self.retPath(graph, start, goal)
        else:               return self.retCost(graph, start, goal)

    def retPath(self, graph:SearchGraph, start:int, goal:List[int]):
        goalCost = (float('inf'), None) #Cost , path
        visited  = set() #visited nodes
        queue    = [[0,[start]]] #[cost so far, path]
        neighbors = self.neighbors(graph.edges)

        while len(queue) > 0:
            queue = sorted(queue, reverse=True)
            if self.verbose: print(f"goalCost: {goalCost}\tqueue:\t{queue}\nvisited: {visited}\n")
            #i didn't reach goal so explore all neighbors
            cost_a, path = queue.pop()
            node_a = path[-1]
            visited.add(node_a)
            for edge in neighbors[node_a]:
                cost_b = cost_a + edge.weight
                if edge.node_b in visited or cost_b >= goalCost[0]: continue #skip explored nodes or overexpensive
                if edge.node_b in goal: 
                    goalCost = (cost_b, path+[edge.node_b])
                    continue
                queue.append([cost_b, path + [edge.node_b]])

        return  goalCost
    
if __name__ == '__main__':
    test = 1
    verbose = True
    showPath = True
    if test == 1:
        graph = SearchGraph(UniformCostSearch(showPath, verbose))
        graph(Edge(0, 1, 2))
        graph(Edge(0, 3, 5))
        graph(Edge(1, 6, 1))
        graph(Edge(2, 1, 4))
        graph(Edge(3, 1, 5))
        graph(Edge(3, 4, 2))
        graph(Edge(3, 6, 6))
        graph(Edge(4, 2, 4))
        graph(Edge(4, 5, 3))
        graph(Edge(5, 2, 6))
        graph(Edge(5, 6, 3))
        graph(Edge(6, 4, 7))

        print(graph)
        start = 0
        goal  = 6
        path = graph.search(start, goal)
        print(f'Minimum cost from start {start} to goal {goal} is\n{path}')
    if test == 2:
        graph = SearchGraph(UniformCostSearch(True, True))
        graph(Edge(0,  1, 5))
        graph(Edge(0,  2, 9))
        graph(Edge(0,  4, 6))
        graph(Edge(1,  2, 3))
        graph(Edge(1, 10, 9))
        graph(Edge(2,  1, 2))
        graph(Edge(2,  3, 1))
        graph(Edge(3,  0, 6))
        graph(Edge(3,  6, 7))
        graph(Edge(3, 11, 5))
        graph(Edge(4,  0, 1))
        graph(Edge(4,  3, 2))
        graph(Edge(4,  5, 2))
        graph(Edge(5, 12, 7))
        graph(Edge(6,  4, 2))
        graph(Edge(6, 12, 8))

        print(graph)
        start = 0
        goals  = [10, 11, 12]
        path = graph.search(start, goals)
        print(f'Minimum cost from [{start}] -> {goals} is **{path[0]}** with path {path[1]}')