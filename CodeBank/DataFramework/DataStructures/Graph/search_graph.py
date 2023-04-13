from .weighted_graph import WeightedGraph
from typing import Union, List

class SearchGraph(WeightedGraph):
    """A wrapper used in uniform_cost and astar algorithm to include the idea of searching a node in a graph"""
    def __init__(self, heuristic):
        super().__init__()
        self.heuristic = heuristic
    
    def search(self, start_node:int, goal_nodes:Union[int, List[int]]):
        if isinstance(goal_nodes, int): goal_nodes = [goal_nodes]
        return self.heuristic(self, start_node, goal_nodes)
