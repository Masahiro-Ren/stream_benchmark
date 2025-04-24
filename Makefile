CXX = icpx

SRC = stream.cpp
EXE = stream.exe

# CXXFLAGS = --std=c++17 -O3 -fopenmp \
# 			-ftree-vectorize \
# 			-march=native
CXXFLAGS = --std=c++17 -O3 -fiopenmp \
           -mcmodel=large \
           -xHost -qopt-streaming-stores=always \
		   -qopt-zmm-usage=high

all: $(EXE)

$(EXE):$(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(EXE)
