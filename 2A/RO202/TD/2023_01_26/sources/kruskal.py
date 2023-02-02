import numpy as np
import graph
import sys

def main():
    ex3_3()
    ex3_4()



def kruskal(g: graph, isMax: bool = False) -> graph:
    # Créer un nouveau graphe contenant les mêmes sommets que g
    tree = graph.Graph(g.nodes)

    # Nombre d'arêtes dans l'arbre
    addedEdges = 0

    # Récupérer toutes les arêtes de g
    edges = g.getEdges()

    # Trier les arêtes par poids croissant
    edges.sort(reverse=isMax)


    for edge in edges:
        # check with edge makes a cycle
        if tree.createACycle(edge) == False:
            tree.addCopyOfEdge(edge)
            addedEdges += 1

    return tree


def ex3_3():
    print('exercice 3.3)')
    g = graph.Graph(np.array(["a", "b", "c", "d", "e", "f", "g"]))
    g.addEdge("a", "b",  1.0)
    g.addEdge("a", "c",  8.0)
    g.addEdge("b", "c",  2.0)
    g.addEdge("b", "d",  5.0)
    g.addEdge("b", "e",  7.0)
    g.addEdge("b", "f",  9.0)
    g.addEdge("c", "d",  4.0)
    g.addEdge("d", "e",  6.0)
    g.addEdge("d", "g", 12.0)
    g.addEdge("e", "f",  8.0)
    g.addEdge("e", "g", 11.0)
    g.addEdge("f", "g", 10.0)

    tree = kruskal(g)
    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")


    tree = kruskal(g, True)
    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")


def ex3_4():
    print('exercice 3.4)')
    g = graph.Graph(np.array(["a", "b", "c", "d", "e", "f", "g", "h"]))
    g.addEdge("a", "b",  9.0)
    g.addEdge("a", "f",  6.0)
    g.addEdge("a", "h",  9.0)
    g.addEdge("f", "e",  1.0)
    g.addEdge("b", "e",  5.0)
    g.addEdge("b", "c",  5.0)
    g.addEdge("b", "d",  8.0)
    g.addEdge("c", "d",  2.0)
    g.addEdge("c", "g",  5.0)
    g.addEdge("g", "e",  3.0)
    g.addEdge("g", "d",  8.0)
    g.addEdge("g", "h",  5.0)
    g.addEdge("h", "d",  7.0)

    tree = kruskal(g)
    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("Pas d'arbre couvrant")


    tree = kruskal(g, True)
    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")



    g = graph.Graph(np.array(["A", "B", "C", "D", "E", "F"]))
    g.addEdge("A", "B",  4.0)
    g.addEdge("A", "C",  3.0)
    g.addEdge("B", "C",  5.0)
    g.addEdge("C", "D",  2.0)
    g.addEdge("C", "F",  5.0)
    g.addEdge("D", "F",  3.0)
    g.addEdge("E", "F",  3.0)
    g.addEdge("E", "D",  4.0)


    tree = kruskal(g)
    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("Pas d'arbre couvrant")


    tree = kruskal(g, True)
    if tree != None:
        # print(tree)
        print(repr(tree))
    else:
        print("no possible tree")






if __name__ == '__main__':
    main()