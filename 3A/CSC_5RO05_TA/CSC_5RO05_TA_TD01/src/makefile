CFLAGS=-Wall -Wextra
SRC=main.cpp lib.cpp
CC=g++
OUTPUT=main
CC_RB=arm-linux-gnueabihf-g++
OUTPUT_RB=raspberry

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUTPUT)
	$(CC_RB) $(CFLAGS) $(SRC) -o $(OUTPUT_RB)

linux:
	$(CC) $(CFLAGS) $(SRC) -o $(OUTPUT)

raspberry:
	$(CC_RB) $(CFLAGS) $(SRC) -o $(OUTPUT_RB)

clean:
	rm -f $(OUTPUT)
	rm -f $(OUTPUT_RB)