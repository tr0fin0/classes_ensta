include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++14
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = CXXFLAGS
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall
CXXFLAGS += -O3 -march=native -Wall
endif

ALL= TestProductMatrix.exe test_product_matrice_blas.exe calcul_pi.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

TestProductMatrix.exe : TestProductMatrix.cpp Matrix.hpp Matrix.cpp ProdMatMat.cpp
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)	

test_product_matrice_blas.exe : test_product_matrice_blas.cpp Matrix.hpp Matrix.cpp
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)	$(BLAS)

calcul_pi.exe : calcul_pi.cpp 
	$(MPICXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"