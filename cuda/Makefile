# Compiler and flags
NVCC = nvcc
CC = gcc
CFLAGS = -I. -Wall
LDFLAGS = -lcudart -L/usr/local/cuda/lib64

# Files
CUDA_SOURCES = inets.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

EXECUTABLE = inets

# Default target
all: $(EXECUTABLE)

# Linking step
$(EXECUTABLE): $(CUDA_OBJECTS) $(C_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# CUDA compilation step
%.o: %.cu
	$(NVCC) -c $< -o $@ -arch=sm_75 -O3 -use_fast_math

# C compilation step
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(CUDA_OBJECTS) $(C_OBJECTS) $(EXECUTABLE)
