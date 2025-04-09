CFLAGS=-Wall -Wextra
SRC_TD4A=td4a.cpp Monitor.cpp Mutex.cpp TimeoutException.cpp timespec.cpp
SRC_TD4B=td4b.cpp Semaphore.cpp Monitor.cpp Mutex.cpp TimeoutException.cpp timespec.cpp
CC=g++
CC_RB=arm-linux-gnueabihf-g++
OUTPUT_TD4A=td4a
OUTPUT_TD4B=td4b
OUTPUT_TD4A_RB=td4a_rb
OUTPUT_TD4B_RB=td4b_rb

all: clean
	$(CC) $(CFLAGS) $(SRC_TD4A) -o $(OUTPUT_TD4A)
	$(CC) $(CFLAGS) $(SRC_TD4B) -o $(OUTPUT_TD4B)
	$(CC_RB) $(CFLAGS) $(SRC_TD4A) -o $(OUTPUT_TD4A_RB)
	$(CC_RB) $(CFLAGS) $(SRC_TD4B) -o $(OUTPUT_TD4B_RB)

td4a:
	$(CC) $(CFLAGS) $(SRC_TD4A) -o $(OUTPUT_TD4A)
	$(CC_RB) $(CFLAGS) $(SRC_TD4A) -o $(OUTPUT_TD4A_RB)

td4b:
	$(CC) $(CFLAGS) $(SRC_TD4B) -o $(OUTPUT_TD4B)
	$(CC_RB) $(CFLAGS) $(SRC_TD4B) -o $(OUTPUT_TD4B_RB)


raspberry:
	$(CC_RB) $(CFLAGS) $(SRC_TD4A) -o $(OUTPUT_TD4A_RB)
	$(CC_RB) $(CFLAGS) $(SRC_TD4B) -o $(OUTPUT_TD4B_RB)

clean:
	rm -f $(OUTPUT_TD4A)
	rm -f $(OUTPUT_TD4B)
	rm -f $(OUTPUT_TD4A_RB)
	rm -f $(OUTPUT_TD4B_RB)