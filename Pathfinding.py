import numpy as np
from queue import PriorityQueue

class Pathfinding:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))  # 0 = walkable, 1 = obstacle
        self.obstacles = obstacles  # List of obstacle positions
        self.update_grid()

    def update_grid(self):
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = 1

    def a_star(self, start, goal):
        open_list = PriorityQueue()
        open_list.put((0, start))  # f, position
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while not open_list.empty():
            current_f, current = open_list.get()
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    open_list.put((f_score[neighbor], neighbor))
                    came_from[neighbor] = current

        return []  # No path found

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        if x > 0: neighbors.append((x - 1, y))
        if x < self.grid_size - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < self.grid_size - 1: neighbors.append((x, y + 1))
        return neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse path to start to goal
