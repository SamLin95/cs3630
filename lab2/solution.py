from __future__ import division
from numpy import matrix
from math import atan2, degrees, sqrt
import time


class Node:
	def __init__(self, name, position):
		self.name = name
		self.neighbors = set([]) 
		self.position = position
		self.visited = False
		self.neighbor_edges = set([])
	
	def visited():
		return self.visited

	def visit():
		self.visited = True

	def set_neighbors(self, neighbors):
		assert isinstance(neighbors, list)
		self.neighbors = neighbors

	def get_degree(self):
		return len(self.neighbor_edges)

	def __hash__(self):
		return self.name.__hash__()

	def __eq__(self, other):
		return isinstance(other, Node) and self.name == other.name

	def __str__(self):
		return self.name

class Edge:
	def __init__(self, node1, node2):
		assert isinstance(node1, Node) and isinstance(node2, Node)
		self.nodes = (min(node1, node2), max(node1, node2))
		self.visited = False

	def get_opposite(self, node):
		if node == self.nodes[0]:
			return self.nodes[1]
		return self.nodes[0]
	
	def __eq__(self, other):
		return isinstance(other, Edge) and self.nodes == other.nodes

	def __str__(self):
		return '(%s, %s)'%(str(self.nodes[0]), str(self.nodes[1]))

	def __hash__(self):
		return self.nodes.__hash__()

class Graph:
	def __init__(self):
		self.vertices = {}
		self.path = []
		self.actions = []
		self.edges = []
		self.euler_path = []
		self.euler_edge_path = []

	def parse_csv(self, file_path):
		parsing_node = True
		with open(file_path, 'r') as f:
			for line in f.readlines():
				if line == '\n':
					parsing_node = False
					continue
				if parsing_node:
					self.parse_nodes(line)
				else:
					self.parse_edges(line)

	def parse_nodes(self, line):
		line = line.rstrip()
		chars = line.split(',')
		node = Node(chars[0], (int(chars[1]), int(chars[2])))
		self.vertices[chars[0]] = node

	def parse_edges(self, line):
		line = line.rstrip()
		chars = line.split(',')
		self.vertices[chars[0]].neighbors.add(self.vertices[chars[1]])
		self.vertices[chars[1]].neighbors.add(self.vertices[chars[0]])
		#also record edges  
		edge = Edge(self.vertices[chars[0]], self.vertices[chars[1]])
		self.edges.append(edge)
		self.vertices[chars[0]].neighbor_edges.add(edge)
		self.vertices[chars[1]].neighbor_edges.add(edge)


	def get_hamiltonian_path(self):
		self.path.append(self.vertices.values()[0])
		self.hamiltonian_path_utility(1)

	def hamiltonian_path_utility(self, pos):
		if len(self.vertices) == len(self.path):
			return True
		for node in self.path[pos - 1].neighbors:
			if node not in self.path:
				self.path.append(node)
				if (self.hamiltonian_path_utility(pos + 1)):
					return True
				self.path.pop()
		return False

	def get_euler_path(self):
		odd_degree = []
		for node in self.vertices:
			if self.vertices[node].get_degree()%2 != 0:
				odd_degree.append(self.vertices[node])
		if len(odd_degree) == 0:
			print "odd degrees are 0"
			start_node = vertices.values()[0]
			self.euler_path.append(start_node)
			return self.get_euler_path_util(1)
		if len(odd_degree) == 2:
			print "two odd degrees"
			self.euler_path.append(odd_degree[0])
			return self.get_euler_path_util(1)
		return False

	def get_euler_path_util(self, pos):
		if len(self.edges) == len(self.euler_edge_path):
			return True
		last_added = self.euler_path[pos - 1]
		print "last_added: " + str(last_added)
		for edge in last_added.neighbor_edges:
			if edge not in self.euler_edge_path:
				self.euler_edge_path.append(edge)
				self.euler_path.append(edge.get_opposite(last_added))
				if (self.get_euler_path_util(pos + 1)):
					return True
				self.euler_edge_path.pop()
				self.euler_path.pop()
		return False

	def get_action_steps(self):
		points = map(lambda x: x.position, self.euler_path)
		last_position = points[0]
		action_steps = []
		for i in xrange(0, len(points) - 1):
			cur_position = points[i]
			next_position = points[i + 1]
			cur_vector = (cur_position[0] - last_position[0], cur_position[1] - last_position[1])
			target_vector = (next_position[0] - cur_position[0], next_position[1] - cur_position[1])
			angle = int(degrees(atan2(target_vector[1], target_vector[0])) - degrees(atan2(cur_vector[1], cur_vector[0]))) % 360
			distance = sqrt(((cur_position[1] - next_position[1]) ** 2 + (cur_position[0] - next_position[0]) ** 2))
			action_steps.append((angle, distance))
			last_position = cur_position
		return action_steps

	


if __name__ == "__main__":
	graph = Graph()
	graph.parse_csv('CS3630_Lab2_Map2.csv.xls')
	graph.get_euler_path()
	raw_input("press any key to start the robot:...")
	from myro import *
	initialize('/dev/tty.Fluke2-07E6-Fluke2')
	steps = graph.get_action_steps()
	print steps
	for step in steps:
		print 'forward by %2.2f'%(step[1])
		print 'turnBy %2.2f'%(step[0])
		turnBy(step[0], 'deg')
		forward(1, step[1])
	# for node in graph.euler_path:
	# 	print node
	# print graph.get_action_steps()








