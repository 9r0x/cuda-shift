CC = nvcc
CFLAGS = -g --dopt on -O2

SRC = $(wildcard *.cu)
BIN = $(patsubst %.cu,bin/%,$(SRC))

all: bin $(BIN)

bin/%: %.cu
	$(CC) $(CFLAGS) $< -o $@

bin:
	mkdir -p bin

clean:
	rm -rf bin