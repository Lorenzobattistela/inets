#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "inets.cuh"

TokenInfo current_token;

TokenInfo get_next_token(const char **input) {
  while (**input == ' ')
    (*input)++; // skip whitespace
  if (**input == '\0')
    return (TokenInfo){END, 0};

  if (**input == '(') {
    (*input)++;
    return (TokenInfo){L_PAREN, 0};
  }
  if (**input == ')') {
    (*input)++;
    return (TokenInfo){R_PAREN, 0};
  }
  if (**input == '+') {
    (*input)++;
    return (TokenInfo){PLUS, 0};
  }
  if (isdigit(**input)) {
    int value = 0;
    while (isdigit(**input)) {
      value = value * 10 + (**input - '0');
      (*input)++;
    }
    return (TokenInfo){DIGIT, value};
  }
  return (TokenInfo){END, 0};
}

void advance(const char **input) { current_token = get_next_token(input); }

void match(Token expected, const char **input) {
  if (current_token.token == expected) {
    advance(input);
  } else {
    printf("Syntax error: expected %d, found %d\n", expected,
           current_token.token);
    exit(1);
  }
}

ASTNode *create_ast_node(Token token, int value, ASTNode *left,
                         ASTNode *right) {
  ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
  node->token = token;
  node->value = value;
  node->left = left;
  node->right = right;
  return node;
}

ASTNode *parse_term(const char **input) {
  ASTNode *node;
  if (current_token.token == DIGIT) {
    node = create_ast_node(DIGIT, current_token.value, NULL, NULL);
    advance(input);
  } else if (current_token.token == L_PAREN) {
    advance(input);
    node = parse_expression(input);
    match(R_PAREN, input);
  } else {
    printf("Syntax error: unexpected token %d\n", current_token.token);
    exit(1);
  }
  return node;
}

ASTNode *parse_expression(const char **input) {
  ASTNode *node = parse_term(input);
  while (current_token.token == PLUS) {
    Token op = current_token.token;
    advance(input);
    ASTNode *right = parse_term(input);
    node = create_ast_node(op, 0, node, right);
  }
  return node;
}

ASTNode *parse(const char *input) {
  advance(&input);
  return parse_expression(&input);
}

void print_ast(ASTNode *node) {
  if (node == NULL)
    return;
  if (node->token == DIGIT) {
    printf("%d", node->value);
  } else if (node->token == PLUS) {
    printf("(");
    print_ast(node->left);
    printf(" + ");
    print_ast(node->right);
    printf(")");
  }
}

void print_token(Token t) {
  switch (t) {
  case DIGIT:
    printf("DIGIT\n");
    break;
  case PLUS:
    printf("PLUS\n");
    break;
  default:
    break;
  }
}

void free_ast(ASTNode *node) {
  if (node == NULL)
    return;
  free_ast(node->left);
  free_ast(node->right);
  free(node);
}

// =============================================================

int cell_counter = 0;
static int gpu_interactions = 0;


// ========================================== CUDA functions

__device__ int cell_counter_c = 0;
__device__ unsigned int to_reduce = 0;

__device__ int create_cell_c(int **arr_net, int **arr_ports, int* cell_types, int cell_type) {
  int cell_id = atomicAdd(&cell_counter_c, 1);
  atomicExch(&cell_types[cell_id], cell_type);
  for (int i = 0; i < MAX_PORTS; i++) {
    arr_net[cell_id][i] = -1;
    arr_ports[cell_id][i] = -1;
  }
  return cell_id;
}

__device__ int zero_cell_c(int **arr_net, int **arr_ports, int *cell_types) {
  return create_cell_c(arr_net, arr_ports, cell_types, ZERO);
}

__device__ int suc_cell_c(int **arr_net, int **arr_ports, int *cell_types) {
  return create_cell_c(arr_net, arr_ports, cell_types, SUC);
}

__device__ int sum_cell_c(int **arr_net, int **arr_ports, int *cell_types) {
  return create_cell_c(arr_net, arr_ports, cell_types, SUM);
}

__device__ void link_c(int **arr_net, int **arr_ports, int *cell_types, int a_id, int a_port, int b_id, int b_port) {
  if (a_id == -1 && b_id != -1) {
    atomicExch(&arr_net[b_id][b_port], -1);
    atomicExch(&arr_ports[b_id][b_port], -1);
  } else if (a_id != -1 && b_id == -1) {
    atomicExch(&arr_net[a_id][a_port], -1);
    atomicExch(&arr_ports[a_id][a_port], -1);
  } else {
    atomicExch(&arr_net[a_id][a_port], b_id);
    atomicExch(&arr_ports[a_id][a_port], b_port);
    atomicExch(&arr_net[b_id][b_port], a_id);
    atomicExch(&arr_ports[b_id][b_port], a_port);
  }
}

__device__ void delete_cell_c(int cell_id, int **arr_net, int **arr_ports, int *cell_types) {
  atomicExch(&cell_types[cell_id], -1);
  for (int i = 0; i < MAX_PORTS; i++) {
    arr_net[cell_id][i] = -1;
    arr_ports[cell_id][i] = -1;
  }
}

__device__ bool is_valid_rule(int rule) {
  return (rule == SUC_SUM) || (rule == ZERO_SUM);
}


__device__ void suc_sum_c(int **arr_net, int **arr_ports, int *cell_types, int suc, int s) {
  int new_suc = suc_cell_c(arr_net, arr_ports, cell_types);
  // link to the left instead of to the right

  // link sum princ port to what was connected to suc aux port
  link_c(arr_net, arr_ports, cell_types, s, 0, arr_net[suc][1], arr_ports[suc][1]);

  // link new suc princ port to what is connected to sum 2 aux
  link_c(arr_net, arr_ports, cell_types, new_suc, 1, arr_net[s][1], arr_ports[s][1]);

  link_c(arr_net, arr_ports, cell_types, new_suc, 0, s, 1);

  delete_cell_c(suc, arr_net, arr_ports, cell_types);

  atomicAdd(&to_reduce, 1);
}

__device__ void zero_sum_c(int **arr_net, int **arr_ports, int *cell_types, int zero, int s) {
  link_c(arr_net, arr_ports, cell_types, arr_net[s][1], arr_ports[s][1], arr_net[s][2], arr_ports[s][2]);

  delete_cell_c(zero, arr_net, arr_ports, cell_types);
  delete_cell_c(s, arr_net, arr_ports, cell_types);

  atomicAdd(&to_reduce, 1);
}

__global__ void reduce_kernel(int *cell_types, int **arr_cell, int **arr_ports) {
  int idx = threadIdx.x;

  if (idx == 0) {
    atomicExch(&to_reduce, 0);
  }
  __syncthreads();

  int *a = arr_cell[idx];
  if (a == NULL) return;

  int a_connected = arr_cell[idx][0];
  if (a_connected == -1) return;

  int *b = arr_cell[a_connected];
  if (b == NULL) return;
  int b_connected = arr_cell[a_connected][0];

  if (b_connected != idx) {
    return;
  }

  // rule was already identified
  // if 1 and 4 are connected and 4 and 1, we want just to apply 1 and 4 reduction
  if (idx > a_connected) return;

  int a_type = cell_types[idx];
  int b_type = cell_types[a_connected];

  if (a_type == -1 || b_type == -1) {
    return;
  }
  int rule = cell_types[idx] + cell_types[a_connected];

  if (rule == SUC_SUM) {
    if (a_type == SUM && b_type == SUC) {
      suc_sum_c(arr_cell, arr_ports, cell_types, a_connected, idx);
    } else if (a_type == SUC && b_type == SUM) {
      suc_sum_c(arr_cell, arr_ports, cell_types, idx, a_connected);
    }
  } else if (rule == ZERO_SUM) {
    if (a_type == SUM && b_type == ZERO) {
      zero_sum_c(arr_cell, arr_ports, cell_types, a_connected, idx);
    } else if (a_type == ZERO && b_type == SUM) {
      zero_sum_c(arr_cell, arr_ports, cell_types, idx, a_connected);
    }
  }
}

void handle_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
      exit(1);
      return;
  }
}

void process(int **arr_cells, int **arr_ports, int *cell_types) {
  // contains what cell is port X connected
  int **d_arr_cells;
  // contains what port is port X connected
  int **d_arr_ports;

  cudaError_t err = cudaMalloc(&d_arr_cells, MAX_CELLS * sizeof(int *));
  handle_cuda_error(err);
  err = cudaMalloc(&d_arr_ports, MAX_CELLS * sizeof(int *));
  handle_cuda_error(err);

  int *h_arr_cell[MAX_CELLS];
  int *h_arr_port[MAX_CELLS];

  for (int i = 0; i < MAX_CELLS; i++) {
    err = cudaMalloc(&h_arr_cell[i], MAX_PORTS * sizeof(int));
    handle_cuda_error(err);
    err = cudaMalloc(&h_arr_port[i], MAX_PORTS * sizeof(int));
    handle_cuda_error(err);

    int* connected_cell = (int *) malloc(MAX_PORTS * sizeof(int));
    int* connected_port = (int *) malloc(MAX_PORTS * sizeof(int));

    int *cell = arr_cells[i];
    int *port = arr_ports[i];
    for (int j = 0; j < MAX_PORTS; j++) {
      if (cell == NULL || port == NULL) {
        connected_cell[j] = -1;
        connected_port[j] = -1;
      } else {
        connected_cell[j] = cell[j];
        connected_port[j] = port[j];
      }
    }
    
    err = cudaMemcpy(h_arr_cell[i], connected_cell, MAX_PORTS * sizeof(int), cudaMemcpyHostToDevice);
    handle_cuda_error(err);
    free(connected_cell);

    err = cudaMemcpy(h_arr_port[i], connected_port, MAX_PORTS * sizeof(int), cudaMemcpyHostToDevice);
    handle_cuda_error(err);
    free(connected_port);
  }

  err = cudaMemcpy(d_arr_cells, h_arr_cell, MAX_CELLS * sizeof(int *), cudaMemcpyHostToDevice);
  handle_cuda_error(err);

  err = cudaMemcpy(d_arr_ports, h_arr_port, MAX_CELLS * sizeof(int *), cudaMemcpyHostToDevice);
  handle_cuda_error(err);

  int *d_cell_types;
  size_t port_conns_size = MAX_CELLS * sizeof(int);

  err = cudaMalloc(&d_cell_types, port_conns_size);
  handle_cuda_error(err);
  err = cudaMemcpy(d_cell_types, cell_types, port_conns_size, cudaMemcpyHostToDevice);
  handle_cuda_error(err);

  int h_found;
  int *d_found;
  err = cudaMalloc(&d_found, sizeof(int));
  handle_cuda_error(err);
  cudaMemset(d_found, 0, sizeof(int));

  err = cudaMemcpyToSymbol(cell_counter_c, &cell_counter, sizeof(int));
  handle_cuda_error(err);

  int threadsPerBlock = MAX_CELLS;
  int blocksPerGrid = 1;

  int maxThreadsPerBlock;
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  if (threadsPerBlock > maxThreadsPerBlock) {
    blocksPerGrid = (MAX_CELLS + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    threadsPerBlock = maxThreadsPerBlock;
  }
  int zero = 0;

  // ==========================Reduce Setup==============================
  h_found = 1;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  while (h_found > 0) {
    err = cudaMemcpyToSymbol(to_reduce, &zero, sizeof(int));
    handle_cuda_error(err);

    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_cell_types, d_arr_cells, d_arr_ports);

    cudaDeviceSynchronize();

    err = cudaMemcpyFromSymbol(&h_found, to_reduce, sizeof(int));
    handle_cuda_error(err);

    gpu_interactions += h_found;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  float seconds = milliseconds / 1000.0;
  printf("GPU elapsed time (s): %f\n", seconds);
  printf("Interactions per second (without setup): %f\n", gpu_interactions / seconds);


  for (int i = 0; i < MAX_CELLS; i++) {
    err = cudaMemcpy(arr_cells[i], h_arr_cell[i], MAX_PORTS * sizeof(int), cudaMemcpyDeviceToHost);
    handle_cuda_error(err);

    err = cudaMemcpy(arr_ports[i], h_arr_port[i], MAX_PORTS * sizeof(int), cudaMemcpyDeviceToHost);
    handle_cuda_error(err);
  }
  err = cudaMemcpy(cell_types, d_cell_types, MAX_CELLS * sizeof(int), cudaMemcpyDeviceToHost);
  handle_cuda_error(err);

  // FREEING MEMORY
  for (int i = 0; i < MAX_CELLS; i++) {
      err = cudaFree(h_arr_cell[i]);
      handle_cuda_error(err);
      err = cudaFree(h_arr_port[i]);
      handle_cuda_error(err);
  }

  cudaFree(d_cell_types);
  cudaFree(d_found);
  cudaFree(d_arr_cells);
  cudaFree(d_arr_ports);
}
// ==============================================================

void print_cell_type(int type) {
  switch (type) {
    case ZERO:
      printf("ZERO");
      break;
    case SUM:
      printf("SUM");
      break;
    case SUC:
      printf("SUC");
      break;
    default:
      printf("Invalid cell type\n");
      break;
  }
  printf("\n");
}

bool is_null_cell(int *cell, int *port) {
  for (int i = 0; i < MAX_PORTS; i++) {
    if (cell[i] != -1) {
      return false;
    }
  }
  return true;
}

int create_cell(int **arr_net, int **arr_ports, int* cell_types, int cell_type) {
  int cell_id = cell_counter++;
  cell_types[cell_id] = cell_type;
  for (int i = 0; i < MAX_PORTS; i++) {
    arr_net[cell_id][i] = -1;
    arr_ports[cell_id][i] = -1;
  }
  return cell_id;
}

int zero_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, ZERO);
  return cell_id;
}

int suc_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, SUC);
  return cell_id;
}

int sum_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, SUM);
  return cell_id;
}


void link(int **arr_net, int **arr_ports, int *cell_types, int a_id, int a_port, int b_id, int b_port) {
  if (a_id == -1 && b_id != -1) {
    arr_net[b_id][b_port] = -1;
    arr_ports[b_id][b_port] = -1;
  } else if (a_id != -1 && b_id == -1) {
    arr_net[a_id][a_port] = -1;
    arr_ports[a_id][a_port] = -1;
  } else {
    arr_net[a_id][a_port] = b_id; 
    arr_ports[a_id][a_port] = b_port;
    arr_net[b_id][b_port] = a_id;
    arr_ports[b_id][b_port] = a_port;
  }
}

int church_encode(int **arr_cells, int **arr_ports, int *cell_types, int num) {
  int zero = zero_cell(arr_cells, arr_ports, cell_types);
  int to_connect_cell = zero;
  int to_connect_port = 0;

  for (int i = 0; i < num; i++) {
    int suc = suc_cell(arr_cells, arr_ports, cell_types);
    link(arr_cells, arr_ports, cell_types, suc, 1, to_connect_cell, to_connect_port);
    to_connect_cell = suc;
    to_connect_port = 0;
  }
  return to_connect_cell;
}

int find_zero_cell(int *cell_types) {
  for (int i = 0; i < MAX_CELLS; i++) {
    int type = cell_types[i];
    if (type == ZERO) {
      return i;
    }
  }
  return -1;
}

int church_decode(int **arr_cells, int **arr_ports, int *cell_types) {
  int zero = find_zero_cell(cell_types);
  if (zero == -1) {
    printf("Not a church encoded number net!\n");
    return -1;
  }

  int val = 0;

  int port_connected_cell = arr_cells[zero][0];

  while (port_connected_cell != -1) {
    port_connected_cell = arr_cells[port_connected_cell][0];
    val++;
  }
  return val;
}

int to_interaction_net(ASTNode *node, int **arr_cells, int **arr_ports, int *cell_types) {
  if (node == NULL) return -1;

  if (node->token == DIGIT) {
    return church_encode(arr_cells, arr_ports, cell_types, node->value);
  } else if (node->token == PLUS) {
    int left_cell_id = to_interaction_net(node->left, arr_cells, arr_ports, cell_types);
    int left_port = 0;
    int right_cell_id = to_interaction_net(node->right, arr_cells, arr_ports, cell_types);
    int right_port = 0;

    if (cell_types[left_cell_id] == SUM) {
      left_port = 2;
    }
    if (cell_types[right_cell_id] == SUM) {
      right_port = 2;
    }

    int sum = sum_cell(arr_cells, arr_ports, cell_types);
    // linking here
    link(arr_cells, arr_ports, cell_types, sum, 0, left_cell_id, left_port);
    link(arr_cells, arr_ports, cell_types, sum, 1, right_cell_id, right_port);
    return sum;
  }
  return -1;
}

void print_net(int **arr_cells, int **arr_ports, int *cell_types) {
  printf("\nNET (\n");
  for (int i = 0; i < MAX_CELLS; i++) {
    int type = cell_types[i];
    int *cell = arr_cells[i];
    int *port = arr_ports[i];

    if (type == -1 || cell == NULL || port == NULL) {
      continue;
    }
    printf("Cell %i ", i);
    print_cell_type(type);
    printf("ports label: [id](connected_cell, connected_port)\n");
    printf("PORTS: (");
    for (int j = 0; j < MAX_PORTS; j++) {
      if (cell[j] == -1 || port[j] == -1) {
        continue;
      }
      printf("[%i] (%i, %i)  ", j, cell[j], port[j]);
    }
    printf("))\n");
  }
}

int main() {
    gpu_interactions = 0;
    const char *in = "(10 + 20) + (30 + 40)" ;

    ASTNode *ast = parse(in);
    // print_ast(ast);
    
    // we will represent the net by two arrays of arrays of ints representing the cell connected_cell and connected_port
    // one with main port connections (should be refactored later) and one with cell types for a given cell
    // ay array[i] where i is the cell id should give out a cell info.
    // we may pre-alloc an array of size MAX_CELLS and also pre_alloc MAX_CELLS cells.
    int **arr_cells = (int **) malloc(MAX_CELLS * sizeof(int *));
    int **arr_ports = (int **) malloc(MAX_CELLS * sizeof(int *));
    int *cell_types = (int *) malloc(MAX_CELLS * sizeof(int));
  
    for (int i = 0; i < MAX_CELLS; i++) {
      arr_cells[i] = (int *)malloc(MAX_PORTS * sizeof(int));
      arr_ports[i] = (int *)malloc(MAX_PORTS * sizeof(int));
      cell_types[i] = -1;
      for (int j = 0; j < MAX_PORTS; j++) {
        arr_cells[i][j] = -1;
        arr_ports[i][j] = -1;
      }
    }

    to_interaction_net(ast, arr_cells, arr_ports, cell_types);

    process(arr_cells, arr_ports, cell_types);

    int val = church_decode(arr_cells, arr_ports, cell_types);
    printf("Decoded value is %i\n", val);
    
    free_ast(ast);
    for (int i = 0; i < MAX_CELLS; i++) {
      free(arr_cells[i]);
      free(arr_ports[i]);
    }
    free(cell_types);
    free(arr_cells);
    free(arr_ports);

    return 0;   
}

