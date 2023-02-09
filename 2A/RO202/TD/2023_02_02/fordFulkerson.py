import numpy as np
import graph
import sys



def main():
    # weights are capabilities
    graph = example()

    # weights are flows
    flow = fordFulkerson(graph, "s", "t")

    print(flow)


def example():
    g = graph.Graph(np.array(["s", "a", "b", "c", "d", "e", "t"]))

    g.addArc("s", "a", 8)
    g.addArc("s", "c", 4)
    g.addArc("s", "e", 6)
    g.addArc("a", "b", 10)
    g.addArc("a", "d", 4)
    g.addArc("b", "t", 8)
    g.addArc("c", "b", 2)
    g.addArc("c", "d", 1)
    g.addArc("d", "b", 2)
    g.addArc("d", "t", 6)
    g.addArc("e", "b", 4)
    g.addArc("e", "d", 2)

    return g



def fordFulkerson(capability: graph, sNode: str = "s", tNode: str = "t") -> graph:

    def getNextNodes(g: graph, index: int) -> list:
        nextNodes = []

        for i in range(g.n):
            if g.adjacency[index][i] != 0:
                nextNodes.append(i)

        return nextNodes

    def getPreviousNodes(g: graph, index: int) -> list:
        previousNodes = []

        for i in range(g.n):
            if g.adjacency[i][index] != 0:
                previousNodes.append(i)

        return previousNodes

    def getConnections(g: graph, index: int) -> dict:
        connections = {}

        for i in range(g.n):
            for j in range(g.n):
                if g.adjacency[i][j] != 0:
                    if   index == j:
                        connections[i] = -1
                    elif index == i:
                        connections[j] = +1

        return connections


    def findNextNode(actualNode: int) -> int:
        nextNodes = getNextNodes(capability, actualNode)
        for node in nextNodes:
            # print(f"{nextNodes}")
            flowMax = capability.adjacency[actualNode][node]
            flowActual = flow.adjacency[actualNode][node]

            isFlowMax = not(flowActual < flowMax)
            isNodeVisited = status[node] != 0

            if isFlowMax == False and isNodeVisited == False:
                status[node] = +1
                weights.append(+(flowMax-flowActual))   # how much to full flow
                return node


        previousNodes = getPreviousNodes(capability, actualNode)
        for node in previousNodes:
            flowMax = capability.adjacency[node][actualNode]
            flowActual = flow.adjacency[node][actualNode]

            isFlowMax = not(flowActual < flowMax)
            isNodeVisited = status[node] != 0

            if isFlowMax == False and isNodeVisited == False:
                status[node] = -1
                weights.append(+(flowActual))
                return node

        return -1


    def updateFlow() -> None:
        minFlow = min(weights)

        for i in range(1, len(path)):
            if   status[path[i]] == +1:    # direct flow
                flow.adjacency[path[i-1]][path[i]] = flow.adjacency[path[i-1]][path[i]] + minFlow
            elif status[path[i]] == -1:    # reverse flow
                flow.adjacency[path[i]][path[i-1]] = flow.adjacency[path[i]][path[i-1]] - minFlow

        return None


    flow = graph.Graph(capability.nodes)   # declare empty graph to store flow

    nNodes = capability.n
    sIndex = capability.indexOf(sNode)
    tIndex = capability.indexOf(tNode)

    while True:
        path   = []
        weights= []
        status = [0] * nNodes

        path.append(sIndex)

        nextNode = 0
        status[sIndex] = +1 # node visited
        while status[tIndex] == 0 and nextNode != -1:
            nextNode = findNextNode(path[-1])

            if nextNode != -1:
                path.append(nextNode)
            elif nextNode != tIndex:
                continue

        updateFlow()
        if status[tIndex] == 0:
            break

    return flow



if __name__ == '__main__':
    main()