def main():
    processors = ['A7', 'A15']
    algorithms = ['Dijkstra', 'BlowFish']
    arrayA7  = [
        [ 0.7114, 0.7407, 0.7544, 0.7605, 0.7818 ],
        [ 0.3359, 0.3376, 0.3405, 0.3468, 0.3479 ]
    ]
    arrayA15 = [
        [ 0.7265, 0.7498, 0.7610, 0.7664, 0.7787 ],
        [ 0.3771, 0.3796, 0.3819, 0.3874, 0.3885 ],
    ]

    A7  = 100
    A15 = 500
    # print(f'{cortex}: {algorithms[0]} {algorithms[1]}')

    for j in range(len(arrayA7[0])):
        print(f'{2**j:2.0f}', end=" & ")
        for i in range(len(arrayA7)):
            if i != len(arrayA7)-1:
                print(f'{arrayA7[i][j]/A7:1.6f}', end=" & ")
            else:
                print(f'{arrayA7[i][j]/A7:1.6f}', end=" & ")

        for i in range(len(arrayA15)):
            if i != len(arrayA15)-1:
                print(f'{arrayA15[i][j]/A15:1.6f}', end=" & ")
            else:
                print(f'{arrayA15[i][j]/A15:1.6f}', end="")
        print("\\\\")

# A7 Dijkstra 0.7114 0.00711400 0.00142280
# A7 Dijkstra 0.7407 0.00740700 0.00148140
# A7 Dijkstra 0.7544 0.00754400 0.00150880
# A7 Dijkstra 0.7605 0.00760500 0.00152100
# A7 Dijkstra 0.7818 0.00781800 0.00156360

# A7 BlowFish 0.3359 00 0.00067180
# A7 BlowFish 0.3376 00 0.00067520
# A7 BlowFish 0.3405 00 0.00068100
# A7 BlowFish 0.3468 00 0.00069360
# A7 BlowFish 0.3479 00 0.00069580

# A15 Dijkstra 0.7265 0.00726500 0.00145300
# A15 Dijkstra 0.7498 0.00749800 0.00149960
# A15 Dijkstra 0.7610 0.00761000 0.00152200
# A15 Dijkstra 0.7664 0.00766400 0.00153280
# A15 Dijkstra 0.7787 0.00778700 0.00155740

# A15 BlowFish 0.3771 0.00377100 0.00075420
# A15 BlowFish 0.3796 0.00379600 0.00075920
# A15 BlowFish 0.3819 0.00381900 0.00076380
# A15 BlowFish 0.3874 0.00387400 0.00077480
# A15 BlowFish 0.3885 0.00388500 0.00077700






if __name__ == "__main__":
    main()