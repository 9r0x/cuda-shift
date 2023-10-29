CC = nvcc
CFLAGS = -g --dopt on -O3 -Xcompiler -fopenmp

SRC = $(wildcard *.cu)
OBJ = $(SRC:%.cu=bin/%.o)
BIN = bin/tiltshift

ifeq ($(debug),1)
	CFLAGS += -DDEBUG
endif

.PHONY: all clean

all: bin $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@

bin/%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

bin:
	mkdir -p bin

clean:
	rm -rf bin