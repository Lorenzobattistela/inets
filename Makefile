# Compiler
CC = gcc

# Compiler flags
CFLAGS = -I. -Wall -O3 -march=native -mtune=native -ffast-math -funroll-loops

# Directories
SRC_DIR = .
OBJ_DIR = obj

# Source files
C_SRC = $(wildcard $(SRC_DIR)/*.c)

# Object files
C_OBJS = $(C_SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Include directories (current directory for headers)
INCLUDES = -I$(SRC_DIR)

# Target executable
TARGET = inets

# Rule to compile all
all: $(TARGET)

# Rule to link the final executable
$(TARGET): $(C_OBJS)
	$(CC) $(C_OBJS) -o $@

# Rule to compile C source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Ensure the obj directory exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean up build files
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

# Phony targets
.PHONY: all clean
