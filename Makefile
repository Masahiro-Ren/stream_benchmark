NXX = nvcc

SRC = stream.cu
EXE = stream.exe

NXXFLAGS= --std=c++17 -O3

$(EXE):$(SRC)
	$(NXX) $(NXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(EXE)