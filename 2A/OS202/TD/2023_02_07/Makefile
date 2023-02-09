include Make_linux.inc

CXXFLAGS = -std=c++11 -fPIC
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
else
CXXFLAGS += -O3 -march=native -Wall
endif

ALL=bucketSort.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB)


help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "    bucketSort.exe : compile bucket sort executable"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"