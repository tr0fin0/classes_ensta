import numpy as np
import bbNode

# Classe représentant l'arbre de résolution d'un branch-and-bound
class BBTree:

    # Meilleure solution entière connue (None si aucune n'a été trouvée) 
    bestSolution = np.array([])

    # Valeur de l'objectif de la meilleure solution trouvée
    bestObjective = 0.0

    # BBNode représentant la racine de l'arbre
    root = None

    # Créé un arbre en fxant sa racine
    def __init__(self, root):
        self.root = root

    # Utiliser l'algorithme de branch-and-bound pour résoudre le problème
    def solve(self):
        self.root.branch(self)

    # Affiche la solution optimale trouvée
    def displaySolution(self):
        
        if self.bestSolution is None:
            print("No feasible integer solution")
        else:

            print("Optimal solution: z = " + f'{self.bestObjective:.2f}' + ", ", end='')

            variables = "("
            values = "("
            
            # For each variable 
            for i in range(len(self.bestSolution)):
                
                # If the variable is not None in the solution 
                if self.bestSolution[i] != 0.0:
                    
                    # Display it 
                    variables += "x" + str(i+1) + ", "
                    values += "%.0f" % self.bestSolution[i] + ", "

            # Remove the last ", " at the end of variables and values  

            if len(variables) > 1:
                variables = variables[0:max(0, len(variables) - 2)]
                values = values[0:max(0, len(values) - 2)]
            
            print(variables, ") = ", values, ")")
    
def ex1():
    A = np.array([[-2, 2], [2, 3], [9, -2]], dtype = float)
    rhs = np.array([7, 18, 36], dtype = float)
    obj = np.array([3, 2], dtype = float)
    isMinimization = False

    root = bbNode.BBNode.create_root(A, rhs, obj, isMinimization)
    return BBTree(root)
    
def isFractional(d): 
   return abs(round(d) - d) > 1E-6 

def main():

    tree = ex1()    
    tree.solve()
    tree.displaySolution()
            
if __name__ == '__main__':
    main()
