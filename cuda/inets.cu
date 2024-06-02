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

Cell* create_cell(int cell_type, int num_aux_ports) {
    Cell *cell = (Cell*)malloc(sizeof(Cell));
    cell->cell_id = cell_counter++;
    cell->type = cell_type;
    cell->num_aux_ports = num_aux_ports;
    cell->ports = (Port*)malloc((num_aux_ports + 1) * sizeof(Port)); // +1 for the main port
    if (cell->ports == NULL) {
      printf("error allocating memory.");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_aux_ports + 1; ++i) {
        cell->ports[i].connected_cell = -1;
        cell->ports[i].connected_port = -1;
    }
    return cell;
}

void delete_cell(Cell **cells, int cell_id) {
    free(cells[cell_id]->ports);
    free(cells[cell_id]);
    cells[cell_id] = NULL;
}

void add_to_net(Cell **net, Cell *cell) {
    net[cell->cell_id] = cell;
}

Cell* zero_cell(Cell **net) {
    Cell *c = create_cell(ZERO, 0);
    add_to_net(net, c);
    return c;
}

Cell* suc_cell(Cell **net) {
    Cell *c = create_cell(SUC, 1);
    add_to_net(net, c);
    return c;
}

Cell* sum_cell(Cell **net) {
    Cell *c = create_cell(SUM, 2);
    add_to_net(net, c);
    return c;
}

void link(Cell **cells, int a, int a_idx, int b, int b_idx) {
    if (a == -1 && b != -1) {
        cells[b]->ports[b_idx].connected_cell = -1;
        cells[b]->ports[b_idx].connected_port = -1;
    } else if (a != -1 && b == -1) {
        cells[a]->ports[a_idx].connected_cell = -1;
        cells[a]->ports[a_idx].connected_port = -1;
    } else {
        cells[a]->ports[a_idx].connected_cell = b;
        cells[a]->ports[a_idx].connected_port = b_idx;
        cells[b]->ports[b_idx].connected_cell = a;
        cells[b]->ports[b_idx].connected_port = a_idx;
    }
}

// ========================================== CUDA functions

__device__ int create_cell_c(int **arr_net, int **arr_ports, int* cell_types, int cell_type, int *cell_count) {
  int cell_id = atomicAdd(cell_count, 1);
  cell_types[cell_id] = cell_type;
  for (int i = 0; i < MAX_PORTS; i++) {
    arr_net[cell_id][i] = -1;
    arr_ports[cell_id][i] = -1;
  }
  return cell_id;
}

__device__ int zero_cell_c(int **arr_net, int **arr_ports, int *cell_types, int *cell_count) {
  int cell_id = create_cell_c(arr_net, arr_ports, cell_types, ZERO, cell_count);
  return cell_id;
}

__device__ int suc_cell_c(int **arr_net, int **arr_ports, int *cell_types, int *cell_count) {
  int cell_id = create_cell_c(arr_net, arr_ports, cell_types, SUC, cell_count);
  return cell_id;
}

__device__ int sum_cell_c(int **arr_net, int **arr_ports, int *cell_types, int *cell_count) {
  int cell_id = create_cell_c(arr_net, arr_ports, cell_types, SUM, cell_count);
  return cell_id;
}

__device__ void link_c(int **arr_net, int **arr_ports, int *cell_types, int a_id, int a_port, int b_id, int b_port) {
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

__device__ void delete_cell_c(int cell_id, int **arr_net, int **arr_ports, int *cell_types) {
  for (int i = 0; i < MAX_PORTS; i++) {
    arr_net[cell_id][i] = -1;
    arr_ports[cell_id][i] = -1;
  }
  cell_types[cell_id] = -1;
}

__device__ void suc_sum_c(int **arr_net, int **arr_ports, int *cell_types, int suc, int s, int *cell_count) {
  int new_suc = suc_cell_c(arr_net, arr_ports, cell_types, cell_count);
  int suc_first_aux_cell = arr_net[suc][1];
  int suc_first_aux_ports = arr_ports[suc][1];
  link_c(arr_net, arr_ports, cell_types, s, 0, suc_first_aux_cell, suc_first_aux_ports);
  link_c(arr_net, arr_ports, cell_types, new_suc, 0, arr_net[s][2], arr_ports[s][2]);
  link_c(arr_net, arr_ports, cell_types, new_suc, 1, s, 2);
  delete_cell_c(suc, arr_net, arr_ports, cell_types);
}

__device__ void zero_sum_c(int **arr_net, int **arr_ports, int *cell_types, int zero, int s, int *cell_count) {
  int sum_aux_first_connected_cell = arr_net[s][1];
  int sum_aux_first_connected_port = arr_ports[s][1];

  int sum_aux_snd_connected_cell = arr_net[s][2];
  int sum_aux_snd_connected_port = arr_ports[s][2];
  link_c(arr_net, arr_ports, cell_types, sum_aux_first_connected_cell, sum_aux_first_connected_port, sum_aux_snd_connected_cell, sum_aux_snd_connected_port);
  delete_cell_c(zero, arr_net, arr_ports, cell_types);
  delete_cell_c(s, arr_net, arr_ports, cell_types);
}

__global__ void reduce_kernel(int *cell_conns, int *cell_types, int *conn_rules, int *cell_count, int **arr_cell, int **arr_ports) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > MAX_CELLS) return;

  int conn = cell_conns[idx];
  int rule = conn_rules[idx];

  if (conn == -1) return;

  // get the main ports
  int *a_connected_cell = arr_cell[idx];
  int *a_connected_port = arr_ports[idx];
  int a_type = cell_types[idx];

  int *b_connected_cell = arr_cell[conn];
  int *b_connected_port = arr_ports[conn];
  int b_type = cell_types[conn];

  if (a_connected_cell == NULL || a_connected_port == NULL) {
    return;
  } else if (b_connected_cell == NULL || b_connected_port == NULL) {
    return;
  }

  if (rule == SUC_SUM) {
    if (a_type == SUM && b_type == SUC) {
      suc_sum_c(arr_cell, arr_ports, cell_types, conn, idx, cell_count);
    } else {
      suc_sum_c(arr_cell, arr_ports, cell_types, idx, conn, cell_count);
    }
  } else if (rule == ZERO_SUM) {
    if (a_type == SUM && b_type == ZERO) {
      zero_sum_c(arr_cell, arr_ports, cell_types, conn, idx, cell_count);
    } else {
      zero_sum_c(arr_cell, arr_ports, cell_types, idx, conn, cell_count);
    }
  }
}

__device__ bool is_valid_rule(int rule) {
  return (rule == SUC_SUM) || (rule == ZERO_SUM);
}

 __global__ void find_reducible_kernel(int *main_port_connections, int *cell_conns, int *cell_types, int *conn_rules, int *found) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= MAX_CELLS) return;

  cell_conns[idx] = -1;
  conn_rules[idx] = -1;

  int main_port_conn = main_port_connections[idx];
  int a_cell_type = cell_types[idx];

  int connection_main = main_port_connections[main_port_conn];
  int b_cell_type = cell_types[main_port_conn];

  int rule = a_cell_type + b_cell_type;
  if(!is_valid_rule(rule)) {
    return;
  }
  // means both cells are connected by their main ports
  if(connection_main == idx) {
    if (idx > main_port_conn) {
      return;
    }

    cell_conns[idx] = main_port_conn;
    conn_rules[idx] = rule;
    if (*found == 0) {
      atomicAdd(found, 1);
    }
  }
  return;
 }

 void handle_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
      return;
  }
}


Cell **process(int **arr_cells, int **arr_ports, Cell** gpu_net, int *main_port_connections, int* cell_types, Cell **net, int *cell_conns, int *conn_rules) {
  // =========================First Find Setup ==========================

  int *d_main_port_connections;
  int *d_cell_types;
  size_t port_conns_size = MAX_CELLS * sizeof(int);

  cudaError_t err = cudaMalloc(&d_main_port_connections, port_conns_size);
  handle_cuda_error(err);
  err = cudaMemcpy(d_main_port_connections, main_port_connections, port_conns_size, cudaMemcpyHostToDevice);
  handle_cuda_error(err);

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

  int threadsPerBlock = cell_counter;
  int blocksPerGrid = 1;

  int maxThreadsPerBlock;
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
  if (threadsPerBlock > maxThreadsPerBlock) {
    blocksPerGrid = (cell_counter + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    threadsPerBlock = maxThreadsPerBlock;
  }

  // first find reducible, we have to start somewhere
  find_reducible_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_main_port_connections, d_cell_conns, d_cell_types, d_conn_rules, d_found);

  err = cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
  handle_cuda_error(err);

  gpu_interactions += h_found;
  // find reducible needs correct main_port_connections and cell_Types. Cell_conns and cell_rules are fresh


  // ==========================Reduce Setup==============================
  
  // contains what cell is port X connected
  int** d_arr_cells;
  // contains what port is port X connected
  int** d_arr_ports;

  err = cudaMalloc(&d_arr_cells, MAX_CELLS * sizeof(int *));
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

    Cell *c = net[i];
    int* connected_cell = (int *) malloc(MAX_PORTS * sizeof(int));
    int* connected_port = (int *) malloc(MAX_PORTS * sizeof(int));

    for (int j = 0; j < MAX_PORTS; j++) {
      if (c == NULL) {
        connected_cell[j] = -1;
        connected_port[j] = -1;
      } else if (j < c->num_aux_ports + 1) {
        connected_cell[j] = c->ports[j].connected_cell;
        connected_port[j] = c->ports[j].connected_port;
      } else {
        connected_cell[j] = -1;
        connected_port[j] = -1;
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

  int *d_cell_count;
  cudaMalloc(&d_cell_count, sizeof(int));
  cudaMemcpy(d_cell_count, &cell_counter, sizeof(int), cudaMemcpyHostToDevice);


  // ==================================================================
  
  int loop = 0;
  while (h_found > 0) {
    loop++;
    // reduces already update d_arr_cells and d_arr_ports
    // we need to update the main_port connections 
    // (easiest step now is convert back to Cell net) and update from there
    // then do the setup for finding reducible again
    int *h_result_cells[MAX_CELLS];
    int *h_result_ports[MAX_CELLS];

    for (int i = 0; i < MAX_CELLS; i++) {
        h_result_cells[i] = (int *)malloc(MAX_PORTS * sizeof(int));
        h_result_ports[i] = (int *)malloc(MAX_PORTS * sizeof(int));
    }

    int **d_result_cells;
    int **d_result_ports;

    threadsPerBlock = MAX_CELLS;
    blocksPerGrid = 1;
    if (threadsPerBlock > maxThreadsPerBlock) {
      blocksPerGrid = (MAX_CELLS + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
      threadsPerBlock = maxThreadsPerBlock;
    }

    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_cell_conns, d_cell_types, d_conn_rules, d_cell_count, d_arr_cells, d_arr_ports);
    cudaDeviceSynchronize();

    // need to copy to result cells
    err = cudaMalloc(&d_result_cells, MAX_CELLS * sizeof(int *));
    handle_cuda_error(err);
    err = cudaMalloc(&d_result_ports, MAX_CELLS * sizeof(int *));
    handle_cuda_error(err);
    err = cudaMemcpy(d_result_cells, d_arr_cells, MAX_CELLS * sizeof(int *), cudaMemcpyDeviceToDevice);
    handle_cuda_error(err);
    err = cudaMemcpy(d_result_ports, d_arr_ports, MAX_CELLS * sizeof(int *), cudaMemcpyDeviceToDevice);
    handle_cuda_error(err);
    for (int i = 0; i < MAX_CELLS; i++) {
        err = cudaMemcpy(h_result_cells[i], h_arr_cell[i], MAX_PORTS * sizeof(int), cudaMemcpyDeviceToHost);
        handle_cuda_error(err);

        err = cudaMemcpy(h_result_ports[i], h_arr_port[i], MAX_PORTS * sizeof(int), cudaMemcpyDeviceToHost);
        handle_cuda_error(err);
    }
    err = cudaMemcpy(cell_types, d_cell_types, MAX_CELLS * sizeof(int), cudaMemcpyDeviceToHost);
    handle_cuda_error(err);

    for(int i = 0; i < MAX_CELLS; i++) {
      int cell_t = cell_types[i];
      if (cell_t == -1) continue;
    }


    // build up new main_port connections
    for (int i = 0; i < MAX_CELLS; i++) {
      main_port_connections[i] = h_result_cells[i][0];
    }

    // prepare find kernel
    // malloc again because we freed before
    err = cudaMalloc(&d_cell_conns, port_conns_size);
    handle_cuda_error(err);
    err = cudaMalloc(&d_conn_rules, port_conns_size);
    handle_cuda_error(err);
    
    // cell_types are already updated
    // main port connections and cell types are updated on update_connections_and_cell_types(), but we need to pass them onto gpu again
    err = cudaMemcpy(d_main_port_connections, main_port_connections, port_conns_size, cudaMemcpyHostToDevice);
    handle_cuda_error(err);

    err = cudaMemcpy(d_cell_types, cell_types, port_conns_size,   cudaMemcpyHostToDevice);
    handle_cuda_error(err);

    cudaMemset(d_found, 0, sizeof(int));

    threadsPerBlock = MAX_CELLS;
    blocksPerGrid = 1;

    if (threadsPerBlock > maxThreadsPerBlock) {
      blocksPerGrid = (cell_counter + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
      threadsPerBlock = maxThreadsPerBlock;
    }

    find_reducible_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_main_port_connections, d_cell_conns, d_cell_types, d_conn_rules, d_found);
    cudaDeviceSynchronize();

    err = cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    handle_cuda_error(err);

    gpu_interactions += h_found;
    if (h_found == 0) {
      from_gpu_to_net(gpu_net, h_result_cells, h_result_ports, cell_types);
    }

    err = cudaMemcpy(cell_conns, d_cell_conns, port_conns_size, cudaMemcpyDeviceToHost);
    handle_cuda_error(err);

    err = cudaMemcpy(conn_rules, d_conn_rules, port_conns_size, cudaMemcpyDeviceToHost);
    handle_cuda_error(err);

    cudaFree(d_cell_conns);
    cudaFree(d_conn_rules);
    cudaFree(d_result_cells);
    cudaFree(d_result_ports);
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
  cudaFree(d_main_port_connections);
  cudaFree(d_cell_types);
  cudaFree(d_cell_conns);
  cudaFree(d_conn_rules);
  cudaFree(d_found);
  cudaFree(d_arr_cells);
  cudaFree(d_arr_ports);
  cudaFree(d_cell_count);

  return gpu_net;
}
// ==============================================================


int church_encode(Cell **net, int num) {
    Cell *zero = zero_cell(net);
    Cell *to_connect_cell = zero;
    int to_connect_port = 0;

    for (int i = 0; i < num; ++i) {
        Cell *suc = suc_cell(net);
        link(net, suc->cell_id, 1, to_connect_cell->cell_id, to_connect_port);
        to_connect_cell = suc;
        to_connect_port = 0;
    }
    return to_connect_cell->cell_id;
}

Cell* find_zero_cell(Cell **net) {
    for (int i = 0; i < MAX_CELLS; ++i) {
        Cell *c = net[i];
        if (c == NULL) continue;
        if (c->type == ZERO) {
          return c;
        }
    }
    return NULL;
}

int church_decode(Cell **net) {
    Cell *cell = find_zero_cell(net);
    if (cell == NULL) {
        printf("Not a church encoded number net!\n");
        return -1;
    }

    int val = 0;

    Port port = cell->ports[0];

    while (port.connected_cell != -1) {
        port = net[port.connected_cell]->ports[0];
        val++;
    }
    return val;
}

int to_net(Cell **net, ASTNode *node) {
    if (!node) return -1;

    if (node->token == DIGIT) {
        return church_encode(net, node->value);
    } else if (node->token == PLUS) {
        int left_cell_id = to_net(net, node->left);
        int left_port = 0;
        int right_cell_id = to_net(net, node->right);
        int right_port = 0;
        Cell *sum = sum_cell(net);

        if (net[left_cell_id]->type == SUM) {
            left_port = 2;
        }

        if (net[right_cell_id]->type == SUM) {
            right_port = 2;
        }

        link(net, sum->cell_id, 0, left_cell_id, left_port);
        link(net, sum->cell_id, 1, right_cell_id, right_port);
        return sum->cell_id;
    }

    return -1;
}

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

void from_gpu_to_net(Cell **net, int **h_result_cells, int **h_result_ports, int *cell_types) {
  cell_counter = 0;
  // h result cells has the port.connected_cell from the cell with cell id i
  // same for port, but connected port.
  for (int i = 0; i < MAX_CELLS; i++) {
    int *cell = h_result_cells[i];
    int *port = h_result_ports[i];
    int cell_type = cell_types[i];
    if (cell == NULL || port == NULL || cell_type == -1 || is_null_cell(cell, port)) { 
      cell_counter++; 
      continue;
    };

    Cell *c;
    if (cell_type == ZERO) {
      c = zero_cell(net);
    } else if(cell_type == SUM) {
      c = sum_cell(net);
    } else if (cell_type == SUC) {
      c = suc_cell(net);
    }
    
    Port *c_ports = (Port *)malloc(MAX_PORTS * sizeof(Port));
    for (int j = 0; j < c->num_aux_ports + 1; j++) {
      c_ports[j].connected_cell = cell[j];
      c_ports[j].connected_port = port[j];
    }
    c->ports = c_ports;
  }

  for (int i = 0; i < MAX_CELLS; i++) {
    Cell *c = net[i];
    if (c == NULL) {
      continue;
    }
  }
}

void update_connections_and_cell_types(Cell **net, int *main_port_connections, int *cell_types) {
  for (int i = 0; i < cell_counter; i++) {
      Cell *c = net[i];
      main_port_connections[i] = -1;
      cell_types[i] = -1;
      if (c == NULL) continue;
      // this way if net[c->ports[0]->connected_cell] main port should be = to i.
      main_port_connections[i] = c->ports[0].connected_cell;
      cell_types[i] = c->type;
    }
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

int to_interaction_net(ASTNode *node, int **arr_cells, int **arr_ports, int *cell_types, int *main_port_connections) {
  if (node == NULL) return -1;

  if (node->token == DIGIT) {
    return church_encode(arr_cells, arr_ports, cell_types, node->value);
  } else if (node->token == PLUS) {
    int left_cell_id = to_interaction_net(node->left, arr_cells, arr_ports, cell_types, main_port_connections);
    int left_port = 0;
    int right_cell_id = to_interaction_net(node->right, arr_cells, arr_ports, cell_types, main_port_connections);
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

// garbage function that goes away as well as main_port_conns
void update_connections_and_cell_types(int **arr_cells, int **arr_ports, int *main_port_connections) {
  for (int i = 0; i < cell_counter; i++) {
      int *cell = arr_cells[i];
      main_port_connections[i] = -1;
      if (cell == NULL) continue;
      // this way if net[c->ports[0]->connected_cell] main port should be = to i.
      main_port_connections[i] = cell[0];
    }
}

int main() {
    Cell *net[MAX_CELLS] = {NULL};
    gpu_interactions = 0;

    const char *in = "((1 + 1) + (1 + 1)) + ((1 + 1) + (1 + 1))";
    ASTNode *ast = parse(in);
    // to_net(net, ast);
    
    // we will represent the net by two arrays of arrays of ints representing the cell connected_cell and connected_port
    // one with main port connections (should be refactored later) and one with cell types for a given cell
    // ay array[i] where i is the cell id should give out a cell info.
    // we may pre-alloc an array of size MAX_CELLS and also pre_alloc MAX_CELLS cells.
    int **arr_cells = (int **) malloc(MAX_CELLS * sizeof(int *));
    int **arr_ports = (int **) malloc(MAX_CELLS * sizeof(int *));
    int *main_port_connections = (int *) malloc(MAX_CELLS * sizeof(int));
    int *cell_types =(int *) malloc(MAX_CELLS * sizeof(int));
  
    for (int i = 0; i < MAX_CELLS; i++) {
      arr_cells[i] = (int *)malloc(MAX_PORTS * sizeof(int));
      arr_ports[i] = (int *)malloc(MAX_PORTS * sizeof(int));
      cell_types[i] = -1;
      main_port_connections[i] = -1;
    }

    church_encode(arr_cells, arr_ports, cell_types, 3);

    int value = church_decode(arr_cells, arr_ports, cell_types);
    printf("decoded val is: %i\n", value);

    // garbage goes away soon
    update_connections_and_cell_types(arr_cells, arr_ports, main_port_connections);

    int *cell_conns = (int *) malloc(MAX_CELLS * sizeof(int));
    int *conn_rules = (int *) malloc(MAX_CELLS * sizeof(int));

    clock_t start, end;
    double time_used;
    Cell *gpu_net[MAX_CELLS] = {NULL};
    start = clock();

    process(arr_cells, arr_ports, gpu_net, main_port_connections, cell_types, net, cell_conns, conn_rules);
    end = clock();

    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    double ips = (double)gpu_interactions / time_used;

    int val = church_decode(gpu_net);
    printf("Decoded value: %d\n", val);

    printf("The program took %f seconds to execute and made %i interactions.\n", time_used, gpu_interactions);
    printf("Interactions per second: %f\n", ips);
    
    for (int i = 0; i < MAX_CELLS; ++i) {
      if (net[i] != NULL) {
          delete_cell(net, i);
      }
    }
    free_ast(ast);

    return 0;   
}

