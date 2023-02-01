import numpy as np

class Edge:

    id1 = 0
    id2 = 0
    weight = 0.0

    def __init__(self, id1, id2, weight):
        self.id1 = id1
        self.id2 = id2
        self.weight = weight
        
    def __lt__(self, other):
        return self.weight <= other.weight

    def __eq__(self, other):
        return self.id1 == other.id1 and self.id2 == other.id2 or self.id1 == other.id2 and self.id2 == other.id1

class Graph:

    n = 0
    nodes = np.array([])
    adjacency = np.empty(0)

    def __init__(self, sNames):
        self.nodes = np.copy(sNames)
        self.n = len(self.nodes)
        self.adjacency = np.zeros((self.n, self.n))

    def addCopyOfEdge(self, edge):
        self.adjacency[edge.id1, edge.id2] = edge.weight
        self.adjacency[edge.id2, edge.id1] = edge.weight

    def addEdge(self, name1, name2, weight):
        id1 = np.where(self.nodes == name1)[0][0]
        id2 = np.where(self.nodes == name2)[0][0]
        self.adjacency[id1, id2] = weight
        self.adjacency[id2, id1] = weight

    def addArc(self, name1, name2, weight):
        id1 = np.where(self.nodes == name1)[0][0]
        id2 = np.where(self.nodes == name2)[0][0]
        self.adjacency[id1, id2] = weight

    def addArcByIndex(self, id1, id2, weight):  
        self.adjacency[id1, id2] = weight

    def getArcs(self):
        arcs = []
        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency[i][j] != 0:
                    arcs.append(Edge(i, j, self.adjacency[i][j]))
        return arcs

    def getEdges(self):
        edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.adjacency[i][j] != 0:
                    edges.append(Edge(i, j, self.adjacency[i][j]))
        return edges
        
    def createACycle(self, edge):
        cycleDetected = False
        reachedNodes = []
        reachedNodes.append(edge.id1)

        if edge.id2 in reachedNodes:
            cycleDetected = True
        else:
            reachedNodes.append(edge.id2)

        nodesToTest = []
        nodesToTest.append(edge.id1)
        nodesToTest.append(edge.id2)

        reachedEdges = []
        reachedEdges.append(Edge(edge.id1, edge.id2, edge.weight))

        while not cycleDetected and len(nodesToTest) > 0:

            currentNode = nodesToTest[0]
            nodesToTest.pop(0)

            neighborIndex = 0

            while not cycleDetected and neighborIndex < self.n:

                currentEdge = Edge(currentNode, neighborIndex, 1)

                # S'il y a une arête
                if self.adjacency[currentNode][neighborIndex] != 0.0 and not currentEdge in reachedEdges:

                    # Si le sommet a déjà été atteint, il y a un cycle
                    if neighborIndex in reachedNodes:
                        cycleDetected = True 

                    # Sinon, ajouter ce sommet à la liste des sommets atteint et à celles des sommets à tester
                    else:
                        reachedNodes.append(neighborIndex)
                        reachedEdges.append(currentEdge)
                        nodesToTest.append(neighborIndex)
                    
                neighborIndex += 1

        return cycleDetected

    """
    Récupère l'identifiant d'un sommet du graphe à partir de son nom
    sName: Nom du sommet
    return: Identifiant du sommet
    """
    def indexOf(self, sName):
        for i in range(len(self.nodes)):
            if self.nodes[i] == sName:
                return i
        
        return -1

    def __repr__(self):

        result = ""
    
        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency[i][j] != 0:
                
                    # S'il y a une arête
                    if self.adjacency[i][j] == self.adjacency[j][i]:
                        
                        # Si c'est la première fois qu'on croise cet arête
                        if i < j:
                            result += repr(self.nodes[i]) + " - " + repr(self.nodes[j]) + " (" + repr(self.adjacency[i][j]) + ")\n" 
    
                    # S'il y a un arc
                    else:
                        result += repr(self.nodes[i]) + " - " + repr(self.nodes[j]) + " (" + repr(self.adjacency[i][j]) + ")\n" 
    
        return result
