import graph
import sys

def main():
    ex4_2()
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



def dijkstra(g, origin):
		
   # Get the index of the origin 
   r = g.indexOf(origin)

   # Next node considered 
   pivot = r
   
   # Liste qui contiendra les sommets ayant été considérés comme pivot
   v2 = []
   v2.append(r)
   
   pred = [0] * g.n
   
   # Les distances entre r et les autres sommets sont initialement infinies
   pi = [sys.float_info.max] * g.n
   pi[r] = 0

   # Ajouter votre code ici 
   # ...

   
if __name__ == '__main__':
    main()
