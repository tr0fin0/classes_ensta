import graph
import sys



def main():
    ex4_2()
    ex4_3()
    ex6()


def dijkstra(g: graph, origin: str) -> graph:
    # creates graph with the same nodes
    tree    = graph.Graph(g.nodes)

    # information from graph
    nNodes = g.n
    unvisitedNodes = list(range(nNodes))

    # function variables
    originIndex = g.indexOf(origin)
    originPred = -1

    # list of nodes already considered as pivots
    pivots = []

    # list of predecessors
    pred = [''] * nNodes
    pred[originIndex] = originPred

    # list of distances from origin to the other nodes
    dist = [sys.float_info.max] * nNodes
    dist[originIndex] = 0


    # need to consider all nodes in graph
    while unvisitedNodes:

        # search node with lowest weight
        currentMin = None
        for node in unvisitedNodes:
            if currentMin == None:
                currentMin = node
            elif dist[node] < dist[currentMin]:
                currentMin = node

        # save neighbors of the lowest weight node
        neighbors = g.getNeighbors(currentMin)

        # update weight with shortest path is found
        for neighbor in neighbors:
            distValue = dist[currentMin] + g.adjacency[currentMin][neighbor]

            if distValue < dist[neighbor]:
                dist[neighbor] = distValue
                pred[neighbor] = currentMin


        unvisitedNodes.remove(currentMin)


    # return shortest path graph
    for i in range(len(pred)):
        if pred[i] == '':
            pass
        else:
            while pred[i] != originPred:
                weight = g.adjacency[pred[i]][i]
                tree.addArcByIndex(pred[i], i, weight)

                i = pred[i]

    return tree




def ex4_2():
    print("exercice 4.2)")
    cities = []
    cities.append("Paris")
    cities.append("Hambourg")
    cities.append("Londres")
    cities.append("Amsterdam")
    cities.append("Edimbourg")
    cities.append("Berlin")
    cities.append("Stockholm")
    cities.append("Oslo")
    cities.append("Rana")

    g = graph.Graph(cities)

    g.addArc("Paris", "Hambourg", 7)
    g.addArc("Paris", "Londres", 4)
    g.addArc("Paris", "Amsterdam", 3)
    g.addArc("Hambourg", "Stockholm", 1)
    g.addArc("Hambourg", "Berlin", 1)
    g.addArc("Londres", "Edimbourg", 2)
    g.addArc("Amsterdam", "Hambourg", 2)
    g.addArc("Amsterdam", "Oslo", 8)
    g.addArc("Amsterdam", "Londres", 1)
    g.addArc("Stockholm", "Oslo", 2)
    g.addArc("Stockholm", "Rana", 5)
    g.addArc("Berlin", "Amsterdam", 2)
    g.addArc("Berlin", "Stockholm", 1)
    g.addArc("Berlin", "Oslo", 3)
    g.addArc("Edimbourg", "Oslo", 7)
    g.addArc("Edimbourg", "Amsterdam", 3)
    g.addArc("Edimbourg", "Rana", 6)
    g.addArc("Oslo", "Rana", 2)

    tree = dijkstra(g, "Paris")
    # tree = dijkstra(g, "Amsterdam")

    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")



def ex4_3():
    print("exercice 4.3)")
    cities = []
    cities.append("a")
    cities.append("b")
    cities.append("c")
    cities.append("d")
    cities.append("e")
    cities.append("f")
    cities.append("g")
    cities.append("r")

    g = graph.Graph(cities)

    g.addArc("r", "a", 5)
    g.addArc("r", "b", 4)
    g.addArc("b", "a", 5)
    g.addArc("b", "c", 3)
    g.addArc("b", "g", 9)
    g.addArc("a", "c", 3)
    g.addArc("d", "a", 8)
    g.addArc("d", "e", 2)
    g.addArc("e", "c", 4)
    g.addArc("c", "d", 2)
    g.addArc("c", "f", 6)
    g.addArc("c", "g", 8)
    g.addArc("g", "f", 5)

    tree = dijkstra(g, "r")
    # tree = dijkstra(g, "Amsterdam")

    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")


    cities = []
    cities.append("A")
    cities.append("B")
    cities.append("C")
    cities.append("D")
    cities.append("E")
    cities.append("F")
    cities.append("G")
    cities.append("r")

    g = graph.Graph(cities)

    g.addArc("r", "A", 2)
    g.addArc("r", "G", 3)
    g.addArc("A", "B", 3)
    g.addArc("A", "F", 1)
    g.addArc("G", "E", 2)
    g.addArc("F", "G", 3)
    g.addArc("F", "D", 4)
    g.addArc("B", "C", 2)
    g.addArc("E", "F", 2)
    g.addArc("E", "D", 3)
    g.addArc("D", "C", 2)

    tree = dijkstra(g, "r")
    # tree = dijkstra(g, "Amsterdam")

    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")



def ex6():
    print("exercice 6) min")
    cities = []
    cities.append("a")
    cities.append("b")
    cities.append("c")
    cities.append("d")
    cities.append("e")

    g = graph.Graph(cities)

    g.addArc("a", "c", 4)
    g.addArc("a", "d", 8)
    g.addArc("d", "c", 6)
    g.addArc("c", "d", 4)
    g.addArc("c", "b", 3)
    g.addArc("b", "d", 2)
    g.addArc("d", "e", 1)
    g.addArc("b", "e", 2)

    tree = dijkstra(g, "a")
    # tree = dijkstra(g, "Amsterdam")

    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")




if __name__ == '__main__':
    main()