# Path problems with Graph algorithms 

############################# Using BFS

from collections import deque

def bfs(graph, start, goal):
    # Create a set to keep track of visited vertices
    visited = set()
    # Create a queue to keep track of vertices to be visited
    queue = deque([(start, [start])])

    # Loop until the queue is empty
    while queue:
        # Pop the first vertex and path from the queue
        (vertex, path) = queue.popleft()
        # Check if the vertex has not been visited
        if vertex not in visited:
            # Mark the vertex as visited
            visited.add(vertex)
            # Loop through the neighbors of the current vertex
            for neighbor in graph[vertex]:
                # Check if the neighbor is the goal vertex
                if neighbor == goal:
                    # Return the path to the goal vertex
                    return path + [neighbor]
                else:
                    # Add the neighbor and its path to the queue
                    queue.append((neighbor, path + [neighbor]))

    # Return None if no path was found
    return None

# Define a graph as a dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Use the bfs function to find the shortest path between vertices 'A' and 'F'
path = bfs(graph, 'A', 'F')
# Print the resulting path
print(path)

################################### Using Depth First Search

def dfs(graph, start, goal):
    # Create a set to keep track of visited vertices
    visited = set()
    # Create a stack to keep track of vertices to be visited
    stack = [(start, [start])]

    # Loop until the stack is empty
    while stack:
        # Pop the last vertex and path from the stack
        (vertex, path) = stack.pop()
        # Check if the vertex has not been visited
        if vertex not in visited:
            # Mark the vertex as visited
            visited.add(vertex)
            # Loop through the neighbors of the current vertex
            for neighbor in graph[vertex]:
                # Check if the neighbor is the goal vertex
                if neighbor == goal:
                    # Return the path to the goal vertex
                    return path + [neighbor]
                else:
                    # Add the neighbor and its path to the stack
                    stack.append((neighbor, path + [neighbor]))

    # Return None if no path was found
    return None

# Define a graph as a dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Use the dfs function to find a path between vertices 'A' and 'F'
path = dfs(graph, 'A', 'F')
# Print the resulting path
print(path)

############################################# Using Dijkstra’s algorithm

import heapq

def dijkstra(graph, start, goal):
    # Create a dictionary to keep track of distances from the start vertex
    distances = {vertex: float('infinity') for vertex in graph}
    # Set the distance from the start vertex to itself to 0
    distances[start] = 0

    # Create a priority queue to keep track of vertices to be visited
    pq = [(0, start)]
    # Loop until the priority queue is empty
    while len(pq) > 0:
        # Pop the vertex with the smallest distance from the priority queue
        current_distance, current_vertex = heapq.heappop(pq)

        # Check if the current distance is greater than the recorded distance
        if current_distance > distances[current_vertex]:
            continue

        # Loop through the neighbors of the current vertex
        for neighbor, weight in graph[current_vertex].items():
            # Calculate the tentative distance to the neighbor
            distance = current_distance + weight

            # Check if the tentative distance is smaller than the recorded distance
            if distance < distances[neighbor]:
                # Update the recorded distance
                distances[neighbor] = distance
                # Add the neighbor and its distance to the priority queue
                heapq.heappush(pq, (distance, neighbor))

    # Return the shortest distance to the goal vertex
    return distances[goal]

# Define a weighted graph as a dictionary of dictionaries
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 4, 'E': 2},
    'C': {'A': 4, 'B': 1, 'F': 3},
    'D': {'B': 4},
    'E': {'B': 2, 'F': 3},
    'F': {'C': 3, 'E': 3}
}

# Use the dijkstra function to find the shortest path between vertices 'A' and 'F'
distance = dijkstra(graph, 'A', 'F')
# Print the resulting shortest distance
print(distance)

################################################ Using Bellman-Ford Algorithm 

def bellman_ford(graph, start, goal):
    # Create a dictionary to keep track of distances from the start vertex
    distances = {vertex: float('infinity') for vertex in graph}
    # Set the distance from the start vertex to itself to 0
    distances[start] = 0

    # Loop V-1 times, where V is the number of vertices in the graph
    for i in range(len(graph) - 1):
        # Loop through all edges in the graph
        for vertex, edges in graph.items():
            for neighbor, weight in edges.items():
                # Calculate the tentative distance to the neighbor
                distance = distances[vertex] + weight
                # Check if the tentative distance is smaller than the recorded distance
                if distance < distances[neighbor]:
                    # Update the recorded distance
                    distances[neighbor] = distance

    # Check for negative cycles
    for vertex, edges in graph.items():
        for neighbor, weight in edges.items():
            # Calculate the tentative distance to the neighbor
            distance = distances[vertex] + weight
            # Check if the tentative distance is smaller than the recorded distance
            if distance < distances[neighbor]:
                # Return None if a negative cycle was found
                return None

    # Return the shortest distance to the goal vertex
    return distances[goal]

# Define a weighted graph as a dictionary of dictionaries
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 4, 'E': 2},
    'C': {'A': 4, 'B': 1, 'F': 3},
    'D': {'B': 4},
    'E': {'B': 2, 'F': 3},
    'F': {'C': 3, 'E': 3}
}

# Use the bellman_ford function to find the shortest path between vertices 'A' and 'F'
distance = bellman_ford(graph, 'A', 'F')
# Print the resulting shortest distance
print(distance)

############################################## Using Floyd-Washall Algorithm
 

def floyd_warshall(graph):
    # Create a dictionary to keep track of distances between all pairs of vertices
    distances = {}
    # Initialize the distances dictionary with the edge weights
    for vertex in graph:
        distances[vertex] = {}
        for neighbor in graph[vertex]:
            distances[vertex][neighbor] = graph[vertex][neighbor]

    # Loop through all pairs of vertices
    for k in graph:
        for i in graph:
            for j in graph:
                # Check if the distance through vertex k is smaller than the recorded distance
                if i in distances and k in distances[i] and j in distances[k]:
                    if j not in distances[i] or distances[i][k] + distances[k][j] < distances[i][j]:
                        # Update the recorded distance
                        distances[i][j] = distances[i][k] + distances[k][j]

    # Return the shortest distances between all pairs of vertices
    return distances

# Define a weighted graph as a dictionary of dictionaries
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 4, 'E': 2},
    'C': {'A': 4, 'B': 1, 'F': 3},
    'D': {'B': 4},
    'E': {'B': 2, 'F': 3},
    'F': {'C': 3, 'E': 3}
}

# Use the floyd_warshall function to find the shortest path between all pairs of vertices
distances = floyd_warshall(graph)
# Print the resulting shortest distances
for vertex, distance in distances.items():
    print(f"{vertex}: {distance}")

################################################## Prim's ALGORITHM

import heapq

def prim(graph):
    # Create a list to keep track of the minimum spanning tree
    mst = []
    # Create a set to keep track of visited vertices
    visited = set()

    # Choose an arbitrary vertex as the starting vertex
    start_vertex = list(graph.keys())[0]
    # Add the starting vertex to the visited set
    visited.add(start_vertex)
    # Create a priority queue to keep track of edges to be visited
    edges = [(weight, start_vertex, neighbor) for neighbor, weight in graph[start_vertex].items()]
    heapq.heapify(edges)

    # Loop until all vertices have been visited
    while len(visited) < len(graph):
        # Pop the edge with the smallest weight from the priority queue
        weight, vertex1, vertex2 = heapq.heappop(edges)
        # Check if the edge connects two unvisited vertices
        if vertex2 not in visited:
            # Add the edge to the minimum spanning tree
            mst.append((vertex1, vertex2, weight))
            # Add the vertex to the visited set
            visited.add(vertex2)
            # Add the edges of the new vertex to the priority queue
            for neighbor, weight in graph[vertex2].items():
                if neighbor not in visited:
                    heapq.heappush(edges, (weight, vertex2, neighbor))

    # Return the minimum spanning tree
    return mst

# Define a weighted, connected graph as a dictionary of dictionaries
graph = {
    'A': {'B': 2, 'C': 3},
    'B': {'A': 2, 'C': 1, 'D': 3},
    'C': {'A': 3, 'B': 1, 'D': 1},
    'D': {'B': 3, 'C': 1}
}

# Use the prim function to find the minimum spanning tree of the graph
mst = prim(graph)
# Print the resulting minimum spanning tree
print(mst)

################################################## Kruskal's Algorithm 

def kruskal(graph):
    # Create a list to keep track of the minimum spanning tree
    mst = []
    # Create a dictionary to keep track of the connected components
    components = {vertex: vertex for vertex in graph}

    # Create a function to find the root of a connected component
    def find(vertex):
        if components[vertex] != vertex:
            components[vertex] = find(components[vertex])
        return components[vertex]

    # Create a function to merge two connected components
    def union(vertex1, vertex2):
        root1 = find(vertex1)
        root2 = find(vertex2)
        if root1 != root2:
            components[root2] = root1

    # Create a list of all edges in the graph
    edges = [(weight, vertex1, vertex2) for vertex1, neighbors in graph.items() for vertex2, weight in neighbors.items()]
    # Sort the edges by weight
    edges.sort()

    # Loop through all edges in the graph
    for weight, vertex1, vertex2 in edges:
        # Check if the edge connects two unconnected components
        if find(vertex1) != find(vertex2):
            # Add the edge to the minimum spanning tree
            mst.append((vertex1, vertex2, weight))
            # Merge the connected components
            union(vertex1, vertex2)

    # Return the minimum spanning tree
    return mst

# Define a weighted, connected graph as a dictionary of dictionaries
graph = {
    'A': {'B': 2, 'C': 3},
    'B': {'A': 2, 'C': 1, 'D': 3},
    'C': {'A': 3, 'B': 1, 'D': 1},
    'D': {'B': 3, 'C': 1}
}

# Use the kruskal function to find the minimum spanning tree of the graph
mst = kruskal(graph)
# Print the resulting minimum spanning tree
print(mst)