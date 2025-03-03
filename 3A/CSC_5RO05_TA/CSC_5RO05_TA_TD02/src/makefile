CFLAGS=-Wall -Wextra
SRC_TD2A=td2a.cpp
SRC_TD2B=td2b.cpp Timer.cpp timespec.cpp
CC=g++
CC_RB=arm-linux-gnueabihf-g++
OUTPUT_TD2A=td2a
OUTPUT_TD2B=td2b
OUTPUT_TD2A_RB=td2a_rb

all:
	$(CC) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A)
	$(CC) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B)

td2a:
	$(CC) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A)

td2b:
	$(CC) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B)

raspberry:
	$(CC_TD2A_RB) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A_RB)

clean:
	rm -f $(OUTPUT_TD2A)
	rm -f $(OUTPUT_TD2B)
	rm -f $(OUTPUT_TD2A_RB)