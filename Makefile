CXX = clang++

SRC = stream.cpp
EXE = stream.exe

CXXFLAGS = --std=c++17 -O3 -fopenmp \
			-Rpass=loop-vectorize \
			-march=native

all: $(EXE)

$(EXE):$(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(EXE)
