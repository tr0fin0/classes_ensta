def speedUp(f: float, s: float) -> float:
    return 1/((1-f) + f/s)

def main():
    speedUp(0.2, 4)
    speedUp(0.5, 2)

main()