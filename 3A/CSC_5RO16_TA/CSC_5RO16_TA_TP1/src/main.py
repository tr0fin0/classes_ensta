"""
 Simple Astar path planning
 
 author: yowlings
 
 modified by David Filliat
"""

import cv2
import numpy as np
import time
import math
import os
from matplotlib import pyplot as plt


# parameters
show_animation = False
heuristic_weight = 1.0

class pathfind():
    """
    Path finding in a occupancy grid map using AStar algorithms
    """
    def __init__(self, map_file, robot_radius, start, goal):
        self.map_file = os.path.abspath(os.path.join(os.getcwd(), 'images', map_file))
        self.robot_radius = robot_radius
        self.start = start
        self.goal = goal
        self.result_img = cv2.imread(self.map_file)
        self.map = cv2.imread(self.map_file, cv2.IMREAD_GRAYSCALE)
        self.columns, self.rows = self.map.shape
        self.threshold()
        self.inflate()

    def distance(self,p,q):
        """
        Euclidian distance, for compatibility, equivalent to math.dist in python 3.8
        """
        return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))

    def threshold(self):
        """
        Binarize the obstacle map to 0/255
        """
        for i in range(self.columns):
            for j in range(self.rows):
                if self.map[i,j]>128:
                    self.map[i,j]=255
                else:
                    self.map[i,j]=0


    def inflate(self):
        """
        Inflate the obstacle size to take the robot radius into account
        """
        old_map = self.map.copy()

        neighbors = np.arange(-self.robot_radius,self.robot_radius+1)
        for i in range(self.columns):
            for j in range(self.rows):
                if old_map[i,j]==0:
                    for u in neighbors:
                        for v in neighbors:
                            if(u!=0 and v!=0):
                                if (((i+u)>=0)&((i+u)<self.columns)&((j+v)>=0)&((j+v)<self.rows)):
                                    self.result_img[i+u,j+v,1]=128
                                    self.map[i+u,j+v]=0


    def get_neighbors(self,current):
        """
        Return neighbor nodes using 8-connectivity
        """
        neighbors = []
        if(current[0]+1<self.columns and self.map[current[0]+1,current[1]]==255):
            neighbors.append((current[0]+1, current[1]))
        if(current[0]-1>=0 and self.map[current[0]-1,current[1]]==255):
            neighbors.append((current[0]-1, current[1]))
        if(current[1]-1>=0 and self.map[current[0],current[1]-1]==255):
            neighbors.append((current[0], current[1]-1))
        if(current[1]+1<self.rows and self.map[current[0],current[1]+1]==255):
            neighbors.append((current[0], current[1]+1))
        if(current[0]+1<self.columns and current[1]+1<self.rows and self.map[current[0]+1,current[1]+1]==255):
            neighbors.append((current[0]+1, current[1]+1))
        if(current[0]-1>=0 and current[1]-1>=0 and self.map[current[0]-1,current[1]-1]==255):
            neighbors.append((current[0]-1, current[1]-1))
        if(current[1]-1>=0 and current[0]+1<self.columns and self.map[current[0]+1,current[1]-1]==255):
            neighbors.append((current[0]+1, current[1]-1))
        if(current[1]+1<self.rows and current[0]-1>=0 and self.map[current[0]-1,current[1]+1]==255):
            neighbors.append((current[0]-1, current[1]+1))

        return neighbors


    def heuristic(self,a,b):
        """
        Return heuristic for AStar
        """
        return self.distance(a,b)


    def draw_path(self):
        """
        Draw the computed path, after AStar
        """
        cv2.circle(self.result_img, (self.start[1],self.start[0]), 2, (255,0,0),2)
        cv2.circle(self.result_img, (self.goal[1],self.goal[0]), 2, (0,0,255),2)

        for current in self.path:
            cv2.circle(self.result_img, (current[1],current[0]), 0, (255,0,0),0)

        return self.result_img

    def get_path_length(self):
        """
        Return path length, after AStar
        """
        length = 0
        for i,k in zip(self.path[0::], self.path[1::]):
            length += self.distance(i, k)
        return length


    def astar(self):
        """
        Compute shortest path using AStar algorithm
        """
        frontier = {}
        frontier[self.start] = heuristic_weight * self.heuristic(self.goal, self.start)

        self.came_from = {}
        cost_so_far = {}
        closed = {}
        self.came_from[self.start] = None
        cost_so_far[self.start] = 0
        iter = 0

        while frontier:
            iter += 1

            current = min(frontier, key=frontier.get)

            # draw frontier for results
            self.result_img[current[0],current[1],1] = 255

            if current == self.goal:
                break

            neighbors = self.get_neighbors(current)
            for next in neighbors:
                new_cost = cost_so_far[current] + self.distance(current, next)
                test = (next in closed) or ((next in frontier) and (new_cost > cost_so_far[next]))
                if not test:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic_weight * self.heuristic(self.goal, next)
                    frontier[next] = priority
                    self.came_from[next] = current

                    # draw frontier for results
                    self.result_img[next[0],next[1],0] = 0
                    self.result_img[next[0],next[1],1] = 0
            closed[current] = 1
            frontier.pop(current)

        # plt.imshow(cv2.distanceTransform(self.map, cv2.DIST_L2, 3))
        # plt.show()
            if show_animation and iter % 100 ==0:
                plt.imshow(self.result_img)
                plt.pause(0.001)

        # Compute path
        current = self.goal
        self.path = [current]
        while current != self.start:
            current = self.came_from[current]
            self.path.append(current)
        self.path.append(self.start)
        self.path.reverse()


def main():
    map_file='office.png'
    start = (100,42)
    goal = (190,150)

    #map_file='labyrinthe.jpg'
    #start = (7,7)
    #goal = (248,248)

    #map_file='freespace.png'
    #start = (40,200)
    #goal = (200,40)


    d1 = pathfind(map_file,2,start,goal)
    begin = time.time()
    d1.astar()
    end = time.time()


    print('Computing time : ',end - begin)
    print('Path length : ', d1.get_path_length())

    result_img = d1.draw_path()

    plt.imshow(result_img)
    plt.savefig('pathfind.png')
    print("Press Q in figure to finish...")
    plt.show()

if __name__ == '__main__':
    main()
