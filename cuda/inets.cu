#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

static int cell_counter = 0;
static int gpu_interactions = 0;


// ========================================== CUDA functions

__device__ void acquire_lock(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // Spin-wait (busy-wait) until the lock is acquired
    }
}

__device__ void release_lock(int *lock) {
    atomicExch(lock, 0);
}

__device__ void acquire_multiple_locks(int *locks, int *cells, int num_cells) {
  for (int i = 0; i < num_cells; i++) {
    acquire_lock(&locks[cells[i]]);
  }
}

__device__ void release_multiple_locks(int *locks, int *cells, int num_cells) {
  for (int i = 0; i < num_cells; i++) {
    release_lock(&locks[cells[i]]);
  }
}

__device__ int create_cell_c(int **arr_net, int **arr_ports, int* cell_types, int cell_type, int *cell_count) {
  int cell_id = atomicAdd(cell_count, 1);
  atomicExch(&cell_types[cell_id], cell_type);
  for (int i = 0; i < MAX_PORTS; i++) {
    atomicExch(&arr_net[cell_id][i], -1);
    atomicExch(&arr_ports[cell_id][i], -1);
  }
  return cell_id;
}

__device__ int zero_cell_c(int **arr_net, int **arr_ports, int *cell_types, int *cell_count) {
  return create_cell_c(arr_net, arr_ports, cell_types, ZERO, cell_count);
}

__device__ int suc_cell_c(int **arr_net, int **arr_ports, int *cell_types, int *cell_count) {
  return create_cell_c(arr_net, arr_ports, cell_types, SUC, cell_count);
}

__device__ int sum_cell_c(int **arr_net, int **arr_ports, int *cell_types, int *cell_count) {
  return create_cell_c(arr_net, arr_ports, cell_types, SUM, cell_count);
}

__device__ void link_c(int **arr_net, int **arr_ports, int *cell_types, int a_id, int a_port, int b_id, int b_port, int *lock) {
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

__device__ void delete_cell_c(int cell_id, int **arr_net, int **arr_ports, int *cell_types, int *lock) {
  for (int i = 0; i < MAX_PORTS; i++) {
    atomicExch(&arr_net[cell_id][i], -1);
    atomicExch(&arr_ports[cell_id][i], -1);
  }
  atomicExch(&cell_types[cell_id], -1);
}

__device__ void suc_sum_c(int **arr_net, int **arr_ports, int *cell_types, int suc, int s, int *cell_count, int *lock) {
  int new_suc = suc_cell_c(arr_net, arr_ports, cell_types, cell_count);

  int suc_first_aux_cell = arr_net[suc][1];
  int suc_first_aux_ports = arr_ports[suc][1];

  link_c(arr_net, arr_ports, cell_types, s, 0, suc_first_aux_cell, suc_first_aux_ports, lock);
  link_c(arr_net, arr_ports, cell_types, new_suc, 0, arr_net[s][2], arr_ports[s][2], lock);
  link_c(arr_net, arr_ports, cell_types, new_suc, 1, s, 2, lock);
  delete_cell_c(suc, arr_net, arr_ports, cell_types, lock);

}

__device__ void zero_sum_c(int **arr_net, int **arr_ports, int *cell_types, int zero, int s, int *cell_count, int *lock) {
  int sum_aux_first_connected_cell = arr_net[s][1];
  int sum_aux_first_connected_port = arr_ports[s][1];

  int sum_aux_snd_connected_cell = arr_net[s][2];
  int sum_aux_snd_connected_port = arr_ports[s][2];

  int cells_to_lock[] = {s, sum_aux_first_connected_cell, sum_aux_snd_connected_cell, zero};

  link_c(arr_net, arr_ports, cell_types, sum_aux_first_connected_cell, sum_aux_first_connected_port, sum_aux_snd_connected_cell, sum_aux_snd_connected_port, lock);
  delete_cell_c(zero, arr_net, arr_ports, cell_types, lock);
  delete_cell_c(s, arr_net, arr_ports, cell_types, lock);
}

__global__ void reduce_kernel(int *cell_conns, int *cell_types, int *conn_rules, int *cell_count, int **arr_cell, int **arr_ports, int *lock) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > MAX_CELLS) return;

  int conn = cell_conns[idx];
  int rule = conn_rules[idx];

  if (conn == -1) return;

  printf("Im thread %i and got here!\n", idx);

  // get the main ports
  int *a_connected_cell = arr_cell[idx];
  int *a_connected_port = arr_ports[idx];
  int a_type = cell_types[idx];

  int *b_connected_cell = arr_cell[conn];
  int *b_connected_port = arr_ports[conn];
  int b_type = cell_types[conn];

  if (a_connected_cell == NULL || a_connected_port == NULL || b_connected_cell == NULL || b_connected_port == NULL || cell_types[idx] == -1 || cell_types[conn] == -1) {
    return;
  }

  printf("Thread %i waiting for lock\n", idx);
  while (atomicCAS(lock, 0, 1) != 0) {
  };
  printf("Thread %i acquired lock!\n", idx);

  __threadfence();

  if (rule == SUC_SUM) {
    printf("SUC SUM with %i and %i\n", idx, conn);
    if (a_type == SUM && b_type == SUC) {
      suc_sum_c(arr_cell, arr_ports, cell_types, conn, idx, cell_count, lock);
    } else {
      suc_sum_c(arr_cell, arr_ports, cell_types, idx, conn, cell_count, lock);
    }
    printf("got out of suc sum!");
  } else if (rule == ZERO_SUM) {
    printf("ZERO SUM with %i and %i\n", idx, conn);
    if (a_type == SUM && b_type == ZERO) {
      zero_sum_c(arr_cell, arr_ports, cell_types, conn, idx, cell_count, lock);
    } else {
      zero_sum_c(arr_cell, arr_ports, cell_types, idx, conn, cell_count, lock);
    }
    printf("got out of zero sum!");
  }

  __threadfence();
  atomicExch(lock, 0);
  printf("Thread %i released lock.\n", idx);
  __threadfence();
}

__device__ bool is_valid_rule(int rule) {
  return (rule == SUC_SUM) || (rule == ZERO_SUM);
}

 __global__ void find_reducible_kernel(int **arr_cells, int **arr_ports, int *cell_conns, int *cell_types, int *conn_rules, int *found, int *lock) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= MAX_CELLS) return;

  atomicExch(&cell_conns[idx], -1);
  atomicExch(&conn_rules[idx], -1);

  __syncthreads();

  int *main_port = arr_cells[idx];
  if (main_port == NULL) return;
  
  int main_port_conn = main_port[0];

  int *connection_port = arr_cells[main_port_conn];
  if (connection_port == NULL) return;
  int connection_main = connection_port[0];

  // printf("Cell at idx %i is connected through main port to %i, which is connected through main port to %i\n", idx, main_port_conn, connection_main);

  if (cell_types[idx] == -1 || cell_types[main_port_conn] == -1) return;

  int rule = cell_types[idx] + cell_types[main_port_conn];
  if(!is_valid_rule(rule)) {
    return;
  }

  __syncthreads();
  if(connection_main == idx) {
    if (idx > main_port_conn) {
      return;
    }

    atomicExch(&cell_conns[idx], main_port_conn);
    atomicExch(&conn_rules[idx], rule);
    printf("set cell_con of %i to %i\n", idx, main_port_conn);
    atomicAdd(found, 1);
  }

  __syncthreads();
  return;
 }

 void handle_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
      return;
  }
}

void process(int **arr_cells, int **arr_ports, int *cell_types) {
  // =========================First Find Setup ==========================
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

  int *d_cell_conns;
  int *d_conn_rules;

  err = cudaMalloc(&d_cell_conns, port_conns_size);
  handle_cuda_error(err);
  err = cudaMalloc(&d_conn_rules, port_conns_size);
  handle_cuda_error(err);

  int h_found;
  int *d_found;
  err = cudaMalloc(&d_found, sizeof(int));
  handle_cuda_error(err);
  cudaMemset(d_found, 0, sizeof(int));

  int threadsPerBlock = MAX_CELLS;
  int blocksPerGrid = 1;

  int maxThreadsPerBlock;
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
  printf("MAX THREADS PER BLOCK: %i\n", maxThreadsPerBlock);
  if (threadsPerBlock > maxThreadsPerBlock) {
    blocksPerGrid = (MAX_CELLS + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    threadsPerBlock = maxThreadsPerBlock;
  }
  printf("USING %i BLOCKS PER GRID\n", blocksPerGrid);

  int *d_lock;
  cudaMalloc(&d_lock, sizeof(int));

  // first find reducible, we have to start somewhere
  find_reducible_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr_cells, d_arr_ports, d_cell_conns, d_cell_types, d_conn_rules, d_found, d_lock);

  cudaMemset(d_lock, 0, sizeof(int));

  err = cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
  handle_cuda_error(err);

  gpu_interactions += h_found;

  printf("H_found is %i\n", h_found);


  // ==========================Reduce Setup==============================

  int *d_cell_count;
  cudaMalloc(&d_cell_count, sizeof(int));
  cudaMemcpy(d_cell_count, &cell_counter, sizeof(int), cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();  
  
  int i = 0;
  while (h_found > 0) {
    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_cell_conns, d_cell_types, d_conn_rules, d_cell_count, d_arr_cells, d_arr_ports, d_lock);
    i++;


    cudaMemset(d_found, 0, sizeof(int));

    cudaDeviceSynchronize();


    if (i == 3) {
      exit(1);
    }

    find_reducible_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr_cells, d_arr_ports, d_cell_conns, d_cell_types, d_conn_rules, d_found, d_lock);

    cudaMemset(d_lock, 0, sizeof(int));

    err = cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    handle_cuda_error(err);

    if (h_found == 0) {
      for (int i = 0; i < MAX_CELLS; i++) {
          err = cudaMemcpy(arr_cells[i], h_arr_cell[i], MAX_PORTS * sizeof(int), cudaMemcpyDeviceToHost);
          handle_cuda_error(err);

          err = cudaMemcpy(arr_ports[i], h_arr_port[i], MAX_PORTS * sizeof(int), cudaMemcpyDeviceToHost);
          handle_cuda_error(err);
      }
      err = cudaMemcpy(cell_types, d_cell_types, MAX_CELLS * sizeof(int), cudaMemcpyDeviceToHost);
      handle_cuda_error(err);
    }
    // print_net(arr_cells, arr_ports, cell_types);
 
    gpu_interactions += h_found;
    
    cudaDeviceSynchronize();
  }

  // FREEING MEMORY
  for (int i = 0; i < MAX_CELLS; i++) {
      cudaFree(h_arr_cell[i]);
      cudaFree(h_arr_port[i]);
  }
  for (int i = 0; i < MAX_CELLS; ++i) {
    cudaFree(h_arr_cell[i]);
    cudaFree(h_arr_port[i]);
  }
  cudaFree(h_arr_cell);
  cudaFree(h_arr_port);
  cudaFree(d_cell_types);
  cudaFree(d_cell_conns);
  cudaFree(d_conn_rules);
  cudaFree(d_found);
  cudaFree(d_arr_cells);
  cudaFree(d_arr_ports);
  cudaFree(d_cell_count);
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
    const char *in = "((10 + 10) + (10 + 10)) + ((10 + 10) + (10 + 10))"; 

    ASTNode *ast = parse(in);
    print_ast(ast);
    
    // we will represent the net by two arrays of arrays of ints representing the cell connected_cell and connected_port
    // one with main port connections (should be refactored later) and one with cell types for a given cell
    // ay array[i] where i is the cell id should give out a cell info.
    // we may pre-alloc an array of size MAX_CELLS and also pre_alloc MAX_CELLS cells.
    int **arr_cells = (int **) malloc(MAX_CELLS * sizeof(int *));
    int **arr_ports = (int **) malloc(MAX_CELLS * sizeof(int *));
    int *cell_types =(int *) malloc(MAX_CELLS * sizeof(int));
  
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

    clock_t start, end;
    double time_used;
  
    start = clock();

    process(arr_cells, arr_ports, cell_types);

    end = clock();

    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    double ips = (double) gpu_interactions / time_used;

    int value = church_decode(arr_cells, arr_ports, cell_types);
    printf("decoded val is: %i\n", value);

    printf("The program took %f seconds to execute and made %i interactions.\n", time_used, gpu_interactions);
    printf("Interactions per second: %f\n", ips);

    print_net(arr_cells, arr_ports, cell_types);
    
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

