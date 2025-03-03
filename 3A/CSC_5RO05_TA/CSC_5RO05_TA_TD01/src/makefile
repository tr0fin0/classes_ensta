CFLAGS=-Wall -Wextra
SRC_TD1A=td1a.cpp timespec.cpp
SRC_TD1B=td1b.cpp timespec.cpp Chrono.cpp
CC=g++
CC_RB=arm-linux-gnueabihf-g++
OUTPUT_TD1A=td1a
OUTPUT_TD1B=td1b
OUTPUT_TD1A_RB=td1a_rb
OUTPUT_TD1B_RB=td1b_rb

all:
	$(CC) $(CFLAGS) $(SRC_TD1A) -o $(OUTPUT_TD1A)
	$(CC) $(CFLAGS) $(SRC_TD1B) -o $(OUTPUT_TD1B)
	$(CC_RB) $(CFLAGS) $(SRC_TD1A) -o $(OUTPUT_TD1A_RB)
	$(CC_RB) $(CFLAGS) $(SRC_TD1B) -o $(OUTPUT_TD1B_RB)

td1a:
	$(CC_TD1A) $(CFLAGS) $(SRC_TD1A) -o $(OUTPUT_TD1A)

td1b:
	$(CC_TD1B) $(CFLAGS) $(SRC_TD1B) -o $(OUTPUT_TD1B)

raspberry:
	$(CC_TD1A_RB) $(CFLAGS) $(SRC_TD1A) -o $(OUTPUT_TD1A_RB)
	$(CC_TD1B_RB) $(CFLAGS) $(SRC_TD1B) -o $(OUTPUT_TD1B_RB)

clean:
	rm -f $(OUTPUT_TD1A)
	rm -f $(OUTPUT_TD1B)
	rm -f $(OUTPUT_TD1A_RB)
	rm -f $(OUTPUT_TD1B_RB)