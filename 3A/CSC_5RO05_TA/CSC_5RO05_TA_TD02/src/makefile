CC=g++
CC_RB=arm-linux-gnueabihf-g++
CFLAGS=-Wall -Wextra

SRC_TD2A=td2a.cpp
SRC_TD2B=td2b.cpp CountDown.cpp Timer.cpp timespec.cpp
SRC_TD2C=td2c.cpp Looper.cpp Timer.cpp timespec.cpp
SRC_TD2D=td2d.cpp Calibrator.cpp Chrono.cpp Looper.cpp Timer.cpp timespec.cpp
SRC_TD2E=td2e.cpp CpuLoop.cpp Calibrator.cpp Chrono.cpp Looper.cpp Timer.cpp timespec.cpp

OUTPUT_TD2A=td2a
OUTPUT_TD2B=td2b
OUTPUT_TD2C=td2c
OUTPUT_TD2D=td2d
OUTPUT_TD2E=td2e

OUTPUT_TD2A_RB=td2a_rb
OUTPUT_TD2B_RB=td2b_rb
OUTPUT_TD2C_RB=td2c_rb
OUTPUT_TD2D_RB=td2d_rb
OUTPUT_TD2E_RB=td2e_rb

all:
	$(CC) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A)
	$(CC) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B)
	$(CC) $(CFLAGS) $(SRC_TD2C) -lrt -o $(OUTPUT_TD2C)
	$(CC) $(CFLAGS) $(SRC_TD2D) -o $(OUTPUT_TD2D)
	$(CC) $(CFLAGS) $(SRC_TD2E) -o $(OUTPUT_TD2E)

td2a:
	$(CC) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A)
	$(CC_RB) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A_RB)

td2b:
	$(CC) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B)
	$(CC_RB) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B_RB)

td2c:
	$(CC) $(CFLAGS) $(SRC_TD2C) -o $(OUTPUT_TD2C)
	$(CC_RB) $(CFLAGS) $(SRC_TD2C) -o $(OUTPUT_TD2C_RB)

td2d:
	$(CC) $(CFLAGS) $(SRC_TD2D) -o $(OUTPUT_TD2D)
	$(CC_RB) $(CFLAGS) $(SRC_TD2D) -o $(OUTPUT_TD2D_RB)

td2e:
	$(CC) $(CFLAGS) $(SRC_TD2E) -o $(OUTPUT_TD2E)
	$(CC_RB) $(CFLAGS) $(SRC_TD2E) -o $(OUTPUT_TD2E_RB)

raspberry:
	$(CC_RB) $(CFLAGS) $(SRC_TD2A) -o $(OUTPUT_TD2A_RB)
	$(CC_RB) $(CFLAGS) $(SRC_TD2B) -o $(OUTPUT_TD2B_RB)
	$(CC_RB) $(CFLAGS) $(SRC_TD2C) -o $(OUTPUT_TD2C_RB)
	$(CC_RB) $(CFLAGS) $(SRC_TD2D) -o $(OUTPUT_TD2D_RB)
	$(CC_RB) $(CFLAGS) $(SRC_TD2E) -o $(OUTPUT_TD2E_RB)

clean:
	rm -f $(OUTPUT_TD2A)
	rm -f $(OUTPUT_TD2B)
	rm -f $(OUTPUT_TD2C)
	rm -f $(OUTPUT_TD2D)
	rm -f $(OUTPUT_TD2E)
	rm -f $(OUTPUT_TD2A_RB)
	rm -f $(OUTPUT_TD2B_RB)
	rm -f $(OUTPUT_TD2C_RB)
	rm -f $(OUTPUT_TD2D_RB)
	rm -f $(OUTPUT_TD2E_RB)