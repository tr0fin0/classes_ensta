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
    g.addArc("e", "t", 2)
    
    return g

# Algorithme of Ford-Fulkerson
def fordFulkerson(inputGraph: graph, startNode: str = "s", finishNode: str = "t") -> graph:
    """
    Marquage des sommets du graphe:
     - mark[i] est égal à +j si le sommet d'indice i peut être atteint en augmentant le flot sur l'arc ji
     - mark[i] est égal à -j si le sommet d'indice i peut être atteint en diminuant  le flot de  l'arc ji
     - mark[i] est égal à sys.float_info.max si le sommet n'est pas marqué
    """
    nNodes = inputGraph.n

    pred   = -1 # -1: no predecessor
                #  i: predecessor i
    signal = +0 # -1: flow to be decrease 
                #  0: no change on flow
                # +1: flow to be increase

    # mark = [pred, signal]
    preds  = [pred]   * nNodes
    status = [signal] * nNodes
    weights = [0] * nNodes

    # marks = np.zeros((nNodes, 2))

    # marks = []
    # for i in range(nNodes):
    #     marks.append(mark)
    # marks = [mark] * nNodes
    # print(f'mark:  {mark}')
    # print(f'marks: {marks}')
    # marked = mark
    # print(marks)
    print(preds)
    print(status)

    # Récupérer l'indice de la source et du puits
    startIndex  = inputGraph.indexOf(startNode)
    finishIndex = inputGraph.indexOf(finishNode)
    # print(f'startIndex: {startIndex}, finishIndex: {finishIndex}')

    # Créer un nouveau graphe contenant les même sommets que g
    flow = graph.Graph(inputGraph.nodes)
    # print(f'flow: {flow}')

    # Récupérer tous les arcs du graphe 
    arcs = inputGraph.getArcs()
    # print(f'arcs: {arcs}')

    # for arc in arcs:
    #     # flow.addArcByIndex(arc.id1, arc.id2, arc.weight)
    #     flow.addArcByIndex(arc.id1, arc.id2, 0)
    #     # flow.addCopyOfEdge(arc)

    # print(repr(flow))
    # print(flow.adjacency)
    # print("flow")

    # for i in range(len(marks)):
    #     print(f'pred: {marks[i][0]} signal: {marks[i][1]}')
        # print(f'{i}: {inputGraph.getNeighbors(i)}')

    # # i = 0
    # print(marks[finishIndex][1])
    # print(signal)
    # # print(finishIndex)
    # # print(len(marks))
    # print(marks[finishIndex][1] == signal)
    while(status[finishIndex] == signal):
        # print("begin 1 while")
        # interation index
        actualIndex = startIndex


        # reset marks to initial state
        # marks = [mark] * nNodes
        # marks = []
        # for i in range(nNodes):
        #     marks.append(mark)
        # print(marks)
        # print(f'p: {preds[actualIndex]} s: {status[actualIndex]}')
        
        # set start index with +
        status[startIndex] = +1
        # marks[startIndex][0] = 1
        # print(marks[startIndex][1])
        # print(marks)


        neighbors = inputGraph.getNeighbors(actualIndex)
        # print(marks)
        print(neighbors)
        # print(status[finishIndex])
        while neighbors and status[finishIndex] != +1:
            
            # print("begin 2 while")
            nextIndex = neighbors[actualIndex]
            # print(nextIndex)
            isMarked_i = status[actualIndex] == +1
            isMarked_j = status[nextIndex] != 0

            flowInt = flow.adjacency[actualIndex][nextIndex]
            flowMax = inputGraph.adjacency[actualIndex][nextIndex]

            isFlowNull = flowInt == 0
            isFlowFull = flowInt < flowMax

            # print(isMarked_i)
            # print(isMarked_j)
            # print(isFlowFull)
            # if marks[actualIndex][1] == +1 and marks[nextIndex][1] == 0 and :
            if isMarked_i and not isMarked_j and isFlowFull:
                # print(f'+i actualIndex: {actualIndex} nextIndex: {nextIndex}')
                status[nextIndex] = +1 # increase flow
                flow.adjacency[actualIndex][nextIndex]
                pred[nextIndex] = actualIndex
                actualIndex = nextIndex

            if not isMarked_i and isMarked_j and not isFlowNull:
                # print(f'-j actualIndex: {actualIndex} nextIndex: {nextIndex}')
                status[nextIndex] = -1 # increase flow
                pred[nextIndex] = actualIndex
                flow.adjacency[][]
                actualIndex = nextIndex

            print(status)
                
            break
        # print("end 2 while")

        break
    # print("end 1 while")


    
    return flow



if __name__ == '__main__':
    main()