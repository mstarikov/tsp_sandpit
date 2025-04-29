from copy import deepcopy

import numpy
import string
import math
from matplotlib import pyplot


class Node:
    def __init__(self, letter, x, y):
        self.letter = letter
        self.x = x
        self.y = y

    @property
    def coordinates(self):
        return self.x, self.y

class Edge:
    def __init__(self, start_node: Node, end_node: Node):
        self.start_node = start_node
        self.end_node = end_node

    def __str__(self):
        return (f'{self.start_node.letter} ({self.start_node.coordinates}) '
                f'-> {self.end_node.letter} ({self.end_node.coordinates}')

    @property
    def distance(self):
        return math.dist(self.start_node.coordinates, self.end_node.coordinates)

    @property
    def midpoint(self):
        return (self.start_node.x + self.end_node.x)/2, (self.start_node.y + self.end_node.y)/2

    @property
    def coordinates(self):
        return self.start_node.coordinates, self.end_node.coordinates

    @property
    def x_points(self):
        return [self.start_node.x, self.end_node.x]

    @property
    def y_points(self):
        return [self.start_node.y, self.end_node.y]

    @property
    def edge_keys(self):
        return self.start_node.letter, self.end_node.letter


class Graph:
    def __init__(self):
        self.coordinates = self.generate_coordinates()
        self.edges = {}
        self.nodes = []
        self.build_nodes_and_edges()

    @staticmethod
    def generate_coordinates(save=True):
        try:
            random_values = numpy.load('graph.pkl', allow_pickle=True)
        except FileNotFoundError:
            random_values = numpy.random.rand(10, 2)
        if save:
            random_values.dump('graph.pkl')
        return random_values

    @property
    def all_node_letters(self):
        return set(node.letter for node in self.nodes)

    def edge_exists(self, edge_key):
        if edge_key in self.edges.keys() or reversed(edge_key) in self.edges.keys():
            return True
        return False

    def add_edge(self, node):
        for existing_node in self.nodes:
            if not self.edge_exists((node.letter, existing_node.letter)):
                self.edges[(node.letter, existing_node.letter)] = Edge(node, existing_node)

    def get_node_edges(self, node):
        return [edge for edge in self.edges.values() if node.letter in edge.edge_keys]

    def get_node_by_letters(self, start_letter, end_letter):
        try:
            return self.edges[(start_letter, end_letter)]
        except KeyError:
            return self.edges[(end_letter, start_letter)]

    def node_properties(self, dimension):
        for node in self.nodes:
            yield getattr(node, dimension)

    def build_nodes_and_edges(self):
        for index, letter in enumerate(string.ascii_uppercase):
            try:
                new_node = Node(letter, *self.coordinates[index])
            except IndexError:
                break
            self.add_edge(new_node)
            self.nodes.append(new_node)

    def show(self):
        for node in self.nodes:
            pyplot.plot(node.x, node.y, 'o')
            pyplot.annotate(node.letter, (node.x, node.y))
        for edge_name, edge in self.edges.items():
            pyplot.plot(edge.x_points, edge.y_points)
            # pyplot.annotate(edge_name, edge.midpoint)
        pyplot.show()

class Path:
    def __init__(self):
        self.edges = []

    def __str__(self):
        return self.edge_keys

    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)

    @property
    def edge_keys(self):
        return [edge.edge_keys for edge in self.edges]

    @property
    def get_odd_nodes(self):
        single_nodes = []
        for node_letter in self.nodes:
            if self.nodes.count(node_letter) == 1:
                single_nodes.append(node_letter)
        return single_nodes

    @property
    def total_distance(self):
        return sum(edge.distance for edge in self.edges)

    @property
    def nodes(self):
        nodes = []
        for edge in self.edges:
            nodes += edge.edge_keys
        return nodes

    @property
    def visited_nodes(self):
        return set(self.nodes)

    def plot(self):
        for node in self.visited_nodes:
            pyplot.plot(node.x, node.y, 'o')
            pyplot.annotate(node.letter, (node.x, node.y))
        for edge in self.edges:
            pyplot.plot(edge.x_points, edge.y_points)
            pyplot.annotate(edge.edge_keys, edge.midpoint)
        pyplot.show()


class Solution:
    def __init__(self, graph):
        self.graph = graph
        self.paths = []

    def add_edge_to_path(self, node = None, path=None):
        node = node or self.graph.nodes[0]
        for edge in self.graph.get_node_edges(node):
            new_path = deepcopy(path) or Path()
            if edge.start_node.letter in new_path.nodes and edge.end_node.letter in new_path.nodes:
                continue
            new_path.add_edge(edge)
            if new_path.visited_nodes == self.graph.all_node_letters:
                new_path.add_edge(self.graph.get_node_by_letters(*path.get_odd_nodes))
                self.paths.append(new_path)
                break
            self.add_edge_to_path(edge.end_node if edge.end_node != node else edge.start_node, new_path)

    def brute_force(self):
        self.add_edge_to_path()
        sorted_by_distance = sorted(self.paths, key=lambda x: x.total_distance)
        # sorted_by_distance[0].plot()
        print(f'shortest path is {sorted_by_distance[0].total_distance} through edges {sorted_by_distance[0].edge_keys}')
        print(f'longest is {sorted_by_distance[-1].total_distance} through edges {sorted_by_distance[-1].edge_keys}')

    def dijkstra(self):
        pass

    def race(self):
        pass


def main():
    graph = Graph()
    solution = Solution(graph)
    solution.brute_force()
    graph.show()

if __name__ == '__main__':
    main()
