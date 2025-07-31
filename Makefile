CXX = nvc++

SRC = stream.cpp
EXE = stream.exe

CXXFLAGS = --std=c++17 -fast -mcmodel=large -mp

all: $(EXE)

$(EXE):$(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(EXE)
