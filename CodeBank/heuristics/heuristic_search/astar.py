

from typing import List, Dict
from CodeBank.DataFramework.DataStructures.Graph import Edge
from CodeBank.DataFramework.DataStructures.Graph import SearchGraph

class ASTAR:
    """A*
    Features:
        - informed search
        - Optimal goal
    
    Requirements:
        - All edge weights (costs) must be POSITIVE
        - Heuristic never overestimates the cost (Admissible Heuristic)

    Parameters:
        - Heuristic, a map from each node to its estimaded node-value for that particular goal
            Example, distance, piece values, 

    Similar to Uniform cost search, but for expansion checks the heuristic added to the cost 
        Ascore = Cost + heuristic

    Sources:
        https://www.youtube.com/watch?v=6TsL96NAZCo
    Complexity:
        O(m^(1+floor(l/e)))
        where
        m: max number of neightbors a node has
        l: length of the shortest path to the goal state
        e: is the least cost of an edge
    """
    def __init__(self, heuristic:Dict[int,int], verbose=False):
        self.verbose = verbose
        self.heuristic = heuristic


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

        goalCost = (float('inf'), None) #Cost , path
        visited  = set() #visited nodes
        queue    = [[0,[start]]] #[cost so far, path]
        neighbors = self.neighbors(graph.edges)
        self.heuristic[start] = 0
        while len(queue) > 0:
            queue = sorted(queue, reverse=True)
            if self.verbose: print(f"goalCost: {goalCost}\tqueue:\t{queue}\nvisited: {visited}\n")
            #i didn't reach goal so explore all neighbors
            cost_a, path = queue.pop()
            node_a = path[-1]
            visited.add(node_a)
            for edge in neighbors[node_a]:
                cost_b = cost_a + edge.weight + self.heuristic[edge.node_b] - self.heuristic[edge.node_a] #undo node_a heuristic
                if edge.node_b in visited or cost_b >= goalCost[0]: continue #skip explored nodes or overexpensive
                if edge.node_b in goal: 
                    goalCost = (cost_b, path+[edge.node_b])
                    continue
                queue.append([cost_b, path + [edge.node_b]])

        return  goalCost
    
if __name__ == '__main__':
    test = 1
    verbose = True
    if test == 1:
        heuristic = {}
        heuristic[0] = 5
        heuristic[1] = 7
        heuristic[2] = 3
        heuristic[3] = 4
        heuristic[4] = 6
        heuristic[5] = 5
        heuristic[6] = 6
        heuristic[10] = 0
        heuristic[11] = 0
        heuristic[12] = 0
        graph = SearchGraph(ASTAR(heuristic, True))
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