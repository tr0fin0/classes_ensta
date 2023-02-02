import numpy as np
import graph
import sys

def main():
    ex3_3()
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

    else:
        print("Pas d'arbre couvrant")




def kruskal(g):
    # Créer un nouveau graphe contenant les mêmes sommets que g
    tree = graph.Graph(g.nodes)

    # Nombre d'arêtes dans l'arbre
    addedEdges = 0
    
    # Récupérer toutes les arêtes de g
    edges = g.getEdges()
    
    # Trier les arêtes par poids croissant
    edges.sort()


    for edge in edges:
        # check with edge makes a cycle
        if tree.createACycle(edge) == False:
            tree.addCopyOfEdge(edge)
            addedEdges += 1

    return tree




if __name__ == '__main__':
    main()