import numpy as np
import sys

class Tableau:

    # Nombre de variables
    n = 0

    # Nombre de contraintes
    m = 0

    A = np.empty(0)
    b = np.array([])
    c = np.array([])

    # Base actuelle ou [] si aucune n'a été définie
    basis = np.array([])

    # Vector de taille n contenant la meilleure solution actuellement connue (ou [] si aucune n'a été trouvée) 
    bestSolution = None

    # Valeur de l'objectif de la meilleure solution connue
    bestObjective = 0

    # Vrai si on considère un problème dont l'objectif est une minimisation
    isMinimization = True

    # Vrai si on souhaite afficher le tableau à chaque itération de l'algorithme
    DISPLAY_SIMPLEX_LOGS = True

    # Crée un tableau
    def __init__(self, A, b, c, isMinimization):

        self.n = len(c)
        self.m = len(A)
        self.A = np.copy(A)
        self.b = np.copy(b)
        self.c = np.copy(c)
        self.isMinimization = isMinimization

        self.basis = np.array([])
        self.bestSolution = None
        self.bestObjective = 0.0

    def ex1(): 

        A = np.array([[1, -1], [0, 1], [8, 5]], dtype = float)

        c = np.array([2, 1], dtype = float)
        b = np.array([4, 8, 56], dtype = float)

        return Tableau(A, b, c, False) 

    def ex2():

        A = np.array([[1, -2, 1, -1, 0, 0], [0, 1, 3, 0, 1, 0], [2, 0, 1, 2, 0, 1]], dtype = float)

        c = np.array([2, -3, 5, 0, 0, 0], dtype = float)
        b = np.array([4, 6, 7], dtype = float)

        return Tableau(A, b, c, True)

    def main():

        # Si le problème n'est pas sous forme normale, il faut le transformer 
        normalForm = False

        # Si on résout un problème sous forme normale 
        if normalForm:

            #** 1er cas - PL Ax = b et une base est fournie (aucune variable d'écart n'est ajoutée au problème) 
            t1 = ex2()
            t1.basis = np.array([0, 2, 5])
            t1.applySimplex()
            
        # Si on résout un problème qui n'est pas sous forme normale 
        else: 

            #** 2ème cas - PL Ax <= b, ajouter des variables d'écart et les utiliser comme base
            t2 = ex1()
            t2.addSlackAndSolve()
            t2.displaySolution()

    # Crée un tableau avec une variable d'écart pour chaque contrainte et résoudre
    def addSlackAndSolve(self):

        # Crée un tableau dans lequel une variable d'écart est ajouté pour chaque contrainte 
        # et sélectionne les variables d'écart comme base
        tSlack = self.tableauWithSlack()

        # Applique l'algorithme du simplexe sur le tableau avec les variables d'écart
        tSlack.applySimplex()

        # Met la solution dans tSlack
        self.setSolution(tSlack)

    # Applique l'algorithme du simplexe
    def applySimplex(self):

        # Affiche le tableau initial
        if self.DISPLAY_SIMPLEX_LOGS:
            print("Tableau initial: ")
            self.display()
        
        # Perturbe chaque valeur de b pour éviter les divisions par zéro quand la base est dégénérée
        eps = 1E-7
        
        for i in range(self.m):
            self.b[i] += eps
            eps *= 0.1

        # Tant que la solution de base peut être améliorée, effectuer un pivotage
        while self.pivot():
            if self.DISPLAY_SIMPLEX_LOGS:
                self.display()

        # Afficher le tableau final
        if self.DISPLAY_SIMPLEX_LOGS:
            print("Final array")
            self.display()

    """
     Effectuer un pivotage. Une base doit avoir été sélectionnée
     Sortie : Vrai si une nouvelle vase a été trouvée, faux si une solution optimale est atteinte
    """ 
    def pivot(self):

        """
         1) Mise sous forme canonique
         *   (rendre la matrice B égale à la matrice unité)
         *
         * Description des variables à utiliser :
         * - A[][] : matrice des contraintes (taille m * n)
         * - b[] : coefficients du membre de droite (taille m)
         * - c[] : coefficients de l'objectif (taille n)
         * - bestObjective : valeur de l'objectif de la solution courante
         * - basis[] : indice des variables dans la base courante (taille m)
         *     - basis[  0] : indice de la première variable dans la base 
         *     - basis[m-1] : indice de la dernière variable de la base.
         *
         * Pseudo-code:
         *
         * l1 - Pour chaque contrainte i (i.e., pour chaque ligne i de A)
         *       l2 - Utiliser une combinaison linéaire de la contrainte i pour fixer à 1 le coefficient en ligne i et en colonne basis[i]
         *       l3 - Utiliser une combinaison linéaire de la contrainte i et des autres contraintes pour fixer les autres coefficients de la colonne basis[i] à 0
         *       l4 - Utiliser une combinaison linéaire de la contrainte i et de c pour fixer c[basis[i]] à 0
         *
         * Remarques :
         * - dans l2 et l3, ne pas oublier de mettre à jour b
         * - dans l4, ne pas oublier de mettre à jour bestObjective
         *              
        """

        # TODO

        # Afficher le tableau sous forme canonique
        if self.DISPLAY_SIMPLEX_LOGS:
            print("Tableau in canonical form")
            self.display()

        # 2 - Obtenir la nouvelle base

        """
         2.1 - Obtenir la variable entrant en base
          
         Indication : Trouver la variable ayant le meilleur coût réduit
          
         Remarque : 
           - Le traitement n'est pas le même si le problème est une maximisation ou une minimisation (utiliser la variable isMinimization)
           - Comme les calculs machine sont approchés, il faut toujours faire des comparaisons numériques à un epsilon prêt. Par exemple :
               - si vous voulez tester si a est supérieur à 1, il faut écrire : a > 1 + epsilon (sinon la condition serait vérifiée pour a = 1.00000000001) 
               - si vous voulez tester si a est inférieur à 1, il faut écrire : a < 1 - epsilon (sinon la condition serait vérifiée pour a = 0.99999999999).
        """  

        # TODO

        """
         2.2 - Obtenir la variable quittant la base
          
         Pseudo-code
             Soit e l'indice de la variable entrant en base (trouvée en 2.1).
             l1 - Déterminer la contrainte i ayant un coefficient positif ou nul en colonne e qui minimise le ratio b[i] / A[i][e]
              l2 - Mettre à jour la base
         
         Remarque : il faut une nouvelle fois faire des comparaisons à epsilon prêt.
        """
         
        # TODO

        # 3 - Retourner vrai si une nouvelle base est trouvée et faux sinon

        # TODO


    # Obtenir la solution du tableau qui est supposé être sous forme canonique
    def getSolution(self):

        self.bestSolution = np.array([0.0] * self.n)

        # For each basic variable, get its value 
        for varBase in range(self.m):
            varId = self.basis[varBase]
            self.bestSolution[varId] = self.b[varBase]

    """
     Fixer la solution du tableau self à celle du tableau tSlack
     tSlack: Tableau contenant la solution
    """
    def setSolution(self, tSlack):

        # Obtenur la solution de tSlack
        tSlack.getSolution()

        self.bestSolution = np.array([0.0] * self.n)

        for varId in range(self.n):
            self.bestSolution[varId] = tSlack.bestSolution[varId]
            print("varId = ", varId, " solution value: ", "%.2f" % tSlack.bestSolution[varId])

        self.bestObjective = tSlack.bestObjective

    # Afficher la solution courante
    def displaySolution(self):

        print("z = ", "%.2f" % self.bestObjective, ", ")

        variables = "("
        values = "("
        for i in range(len(self.bestSolution)):
            if self.bestSolution[i] != 0.0:
                variables += "x" + str(i+1) + ", "

                if isFractional(self.bestSolution[i]):
                    values += str("%.2f" % self.bestSolution[i]) + ", "
                else:
                    values += str("%.2f" % self.bestSolution[i]) + ", "

        variables = variables[0:max(0, len(variables) - 2)]
        values = values[0:max(0, len(values) - 2)]
        print(variables, ") = ", values, ")")

    """
     * Crée un tableau avec une variable d'écart pour chaque contrainte et utilise ces variables d'écart comme base
     * Sortie: Un tableau comportant n+m variables (les n d'origine + m variables d'écart)
    """
    def tableauWithSlack(self):

        ASlack = np.zeros((self.m, self.n+self.m))

        # Pour chaque contrainte
        for cstr in range(self.m):

            # Fixer les coefficients des n variables d'origine
            for col in range(self.n):
                ASlack[cstr][col] = self.A[cstr][col]

            # Fixer le coefficient de la variable de slack non nulle
            ASlack[cstr][self.n + cstr] = 1.0

        # Augmenter le nombre de variables dans l'objectif
        cSlack = np.array([0.0] * (self.n + self.m))

        for i in range(self.n):
            cSlack[i] = self.c[i]

        # Créer une base avec les variables d'écart
        self.basis = np.array([0] * self.m)

        for i in range(self.m):
            self.basis[i] = i + self.n

        slackTableau = Tableau(ASlack, self.b, cSlack, self.isMinimization)
        slackTableau.basis = self.basis

        return slackTableau

    # Afficher le tableau
    def display(self):

        toDisplay = "\nVar.\t"

        for i in range(self.n):
            toDisplay += "x" + str(i+1) + "\t"

        dottedLine = ""
        for i in range(self.n + 2):
            dottedLine += "--------"

        print(toDisplay, "  (RHS)\t\n", dottedLine)

        for l in range(self.m):

            toDisplay = "(C" + str(l+1) + ")\t"

            for c in range(self.n):
                toDisplay += str("%.2f" % self.A[l][c]) + "\t"
            print(toDisplay, "| ",  "%.2f" % self.b[l])

        print(dottedLine)
        toDisplay = "(Obj)\t"

        for i in range(self.n):
            toDisplay += str("%.2f" % self.c[i]) + "\t"

        print(toDisplay, "|  ", "%.2f" % self.bestObjective)

        # Si un solution a été calculée
        if len(self.basis) > 0:
            print(dottedLine)
            self.getSolution()
            self.displaySolution()
        print()

    """
     Créer le tableau utilisé pour la phase 1 de l'algorithme du simplexe ainsi que la base correspondante
     Sortie: Un tableau contenant une variable additionnelle pour chaque contrainte ayant un second membre négatif et fixe les coefficients de l'objectifs correspondant à ceux du simplexe phase 1)

    Utile pour le TP du chapitre 4 sur le branch-and-bound 
    """
    def tableauPhase1(self, negativeRHSCount):

        tSlack = self.tableauWithSlack()

        cPhase1 = np.array([0.0] * (tSlack.n + negativeRHSCount))
        APhase1 = np.zeros((self.m, tSlack.n + negativeRHSCount))

        negativeId = 0

        # Pour chaque contrainte
        for i in range(self.m):

            for j in range(tSlack.n):
                APhase1[i][j] = tSlack.A[i][j]

            # Si le second membre est négatif, ajouter une variable d'écart
            if self.b[i] < -1E-6:
                APhase1[i][tSlack.n + negativeId] = -1.0
                cPhase1[tSlack.n + negativeId] = -1
                negativeId += 1
                
        # Créer le nouveau tableau
        sPhase1 = Tableau(APhase1, self.b, cPhase1, False)

        # Fixer la base
        negativeId = 0

        sPhase1.basis = np.array([0] * self.m)

        for i in range(self.m):
            if self.b[i] < -1E-6:
                sPhase1.basis[i] = tSlack.n + negativeId
                negativeId += 1
            else:
                sPhase1.basis[i] = i + self.n

        return sPhase1
    
    # Appliquer l'algorithme du simplexe phase 1 et 2
    # Utile pour le TP du chapitre 4 sur le branch-and-bound 
    def applySimplexPhase1And2(self):

        tSlack = self.tableauWithSlack()

        # Compter le nombre de contraintes ayant un second membre négatif  
        negativeRHS = 0

        for i in range(self.m):
            if self.b[i] < -1E-6:
                negativeRHS += 1

        isInfeasible = False

        # Si le vecteur 0 n'est pas une solution réalisable  
        if negativeRHS > 0:

            tPhase1 = self.tableauPhase1(negativeRHS)

            if self.DISPLAY_SIMPLEX_LOGS:
                print("\nInitial array: ")
                tPhase1.display()

            while tPhase1.pivot():
                pass # Instruction qui ne fait rien

            if self.DISPLAY_SIMPLEX_LOGS:
                print("Final array")
                tPhase1.display()

            # Si aucune solution du problème n'a été trouvée
            if tPhase1.bestObjective < -1E-6:
                isInfeasible = True

            # Si une solution faisable est trouvée
            else:

                # Mettre à jour A et b de tSlack
                for cstr in range(tSlack.m):

                    tSlack.b[cstr] = tPhase1.b[cstr]

                    for var in range(tSlack.n):
                        tSlack.A[cstr][var] = tPhase1.A[cstr][var]

                # Mettre à jour la base de tSlack
                tSlack.basis = tPhase1.basis

                # Tester si toutes les variables de la base sont des variables de tSlack (et pas des variables ajoutées en phase 1) 
                for i in range(tSlack.m):

                    # Si base[i] n'est pas une variable de tSlack
                    if tSlack.basis[i] >= tSlack.n:

                        # Trouver une variable qui peut remplacer base[i] dans la base
                        var = 0
                        found = False

                        # Tant qu'une telle variable n'a pas été trouvée
                        while not found and var < tSlack.n:

                            # Si la variable var a un coefficient non nulle dans la contrainte correspondante
                            if abs(tSlack.A[i][var]) > 1E-6:

                                # Si la variable var n'est pas déjà dans la base
                                found = True

                                for j in range(self.m):
                                    if tSlack.basis[i] == var:
                                        found = False

                                if found:
                                    tSlack.basis[i] = var

                            var += 1

        if not isInfeasible:

            if self.DISPLAY_SIMPLEX_LOGS:
                toDisplay = "Base: "

                for i in range(tSlack.m):
                    toDisplay += str(tSlack.basis[i]+1) + ", "
                print(toDisplay)

            tSlack.applySimplex()
            self.setSolution(tSlack)


def isFractional(d): 
    return abs(round(d) - d) > 1E-6

            
if __name__ == '__main__':
    main()
