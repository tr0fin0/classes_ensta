CFLAGS=-Wall -Wextra
SRC_TD2A=td2a.cpp
SRC_TD2B=td2b.cpp CountDown.cpp Timer.cpp timespec.cpp
SRC_TD2C=td2c.cpp Looper.cpp Timer.cpp timespec.cpp
CC=g++
CC_RB=arm-linux-gnueabihf-g++
OUTPUT_TD2A=td2a
OUTPUT_TD2B=td2b
OUTPUT_TD2C=td2c
OUTPUT_TD2A_RB=td2a_rb

all:
	$(CC) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A)
	$(CC) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B)
	$(CC) $(CFLAGS) $(SRC_TD2C) -lrt -o $(OUTPUT_TD2C)

td2a:
	$(CC) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A)

td2b:
	$(CC) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B)

td2c:
	$(CC) $(CFLAGS) $(SRC_TD2C) -o $(OUTPUT_TD2C)

raspberry:
	$(CC_TD2A_RB) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A_RB)

clean:
	rm -f $(OUTPUT_TD2A)
	rm -f $(OUTPUT_TD2B)
	rm -f $(OUTPUT_TD2C)
	rm -f $(OUTPUT_TD2A_RB)