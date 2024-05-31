#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
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

Cell* create_cell(int cell_type, int num_aux_ports) {
    Cell *cell = (Cell*)malloc(sizeof(Cell));
    cell->cell_id = cell_counter++;
    cell->type = cell_type;
    cell->num_aux_ports = num_aux_ports;
    cell->ports = (Port*)malloc((num_aux_ports + 1) * sizeof(Port)); // +1 for the main port
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

void suc_sum(Cell **cells, Cell *suc, Cell *s) {
    Cell *new_suc = suc_cell(cells);
    Port suc_first_aux = suc->ports[1];

    link(cells, s->cell_id, 0, suc_first_aux.connected_cell, suc_first_aux.connected_port);
    link(cells, new_suc->cell_id, 0, s->ports[2].connected_cell, s->ports[2].connected_port);
    link(cells, new_suc->cell_id, 1, s->cell_id, 2);

    delete_cell(cells, suc->cell_id);
}

void zero_sum(Cell **cells, Cell *zero, Cell *s) {
    link(cells, s->ports[1].connected_cell, s->ports[1].connected_port, s->ports[2].connected_cell, s->ports[2].connected_port);
    delete_cell(cells, zero->cell_id);
    delete_cell(cells, s->cell_id);
}

int check_rule(Cell *cell_a, Cell *cell_b, ReductionFunc *reduction_func) {
    if (cell_a == NULL || cell_b == NULL) {
        return 0;
    }
    int rule = cell_a->type + cell_b->type;

    if (rule == SUC_SUM) {
        *reduction_func = suc_sum;
        return 1;
    } else if (rule == ZERO_SUM) {
        *reduction_func = zero_sum;
        return 1;
    }
    return 0;
}

// ========================================== CUDa functions
// __global__ void find_reducible_kernel(Cell** net, ReductionFunc *reduction_func, int *a_id, int *b_id);

void handle_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
      return;
  }
}

__device__ bool is_valid_rule(int rule) {
  return (rule == SUC_SUM) || (rule == ZERO_SUM);
}

 __global__ void find_reducible_kernel(int *main_port_connections, int *cell_conns, int *cell_types, int *conn_rules) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= 3500) return;

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
    cell_conns[idx] = main_port_conn;
    cell_conns[main_port_conn] = idx;
    conn_rules[idx] = rule;
    conn_rules[main_port_conn] = rule;
  }
  return;
 }


// structs are not a good way of interacting with the gpu. Lets think abt an alternative. What info do we need to find active pairs?
// we need the main ports of each cell connections, and who theyre connected to. Both integers. 
// so in theory if we have an array 

// instead of using structs we build 2 arrays
// one that holds main port connections: e.g if we have cells[0] connected to cells[1] by main port, we would have [1, 0]
// and another array holding the rule for them to reduce
// e.g if 1 and 0 are a suc_sum, we would have [1, 1]
// then we dont usre structs on the gpu because of the overhead of copying
// after that we have the data needed to reduce cells.
//  we can turn the reduce and link functions device functions, use atomic operations for linking and try to do all the reduction steps
// at the gpu. But we have to think a little bit more on how to copy aux ports and so on.

// for now try to  make find_reducible_c return two arrays of ints. One containing the cells main port connections and other containing what rule we should use to reduce them
void find_reducible_c(int *main_port_connections, int* cell_types, int *cell_conns, int *conn_rules) {
  int* d_main_port_connections;
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

  int threadsPerBlock = 256;
  int blocksPerGrid = (MAX_CELLS + threadsPerBlock - 1) / threadsPerBlock;

  find_reducible_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_main_port_connections, d_cell_conns, d_cell_types, d_conn_rules);

  err = cudaMemcpy(cell_conns, d_cell_conns, port_conns_size, cudaMemcpyDeviceToHost);
  handle_cuda_error(err);

  err = cudaMemcpy(conn_rules, d_conn_rules, port_conns_size, cudaMemcpyDeviceToHost);
  handle_cuda_error(err);

  for(int i = 0; i < 50; i++) {
    printf("i=%i cell cons=%i, conn_rule=%i\n", i, cell_conns[i], conn_rules[i]);
  }

  cudaFree(d_main_port_connections);
  cudaFree(d_cell_types);
  cudaFree(d_cell_conns);
  cudaFree(d_conn_rules);
}
// ==============================================================


int find_reducible(Cell **cells, ReductionFunc *reduction_func, int *a_id, int *b_id) {
    for (int i = 0; i < cell_counter; ++i) {
        if (cells[i] == NULL) continue;
        Cell *cell = cells[i];
        Port main_port = cell->ports[0];

        if (main_port.connected_port == 0) {
            if (check_rule(cell, cells[main_port.connected_cell], reduction_func)) {
                *a_id = cell->cell_id;
                *b_id = main_port.connected_cell;
                return 1;
            }
        }
    }
    return 0;
}

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
    for (int i = 0; i < cell_counter; ++i) {
        if (net[i] != NULL && net[i]->type == ZERO) {
            return net[i];
        }
    }
    return NULL;
}

int church_decode(Cell **net) {
    Cell *cell = find_zero_cell(net);
    if (!cell) {
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

int main() {
    Cell *net[MAX_CELLS] = {NULL};
    const char *in = "((1 + 1) + (1 + 1)) + ((1 + 1) + (1 + 1))";
    ASTNode *ast = parse(in);
    // print_ast(ast);
    to_net(net, ast);

    ReductionFunc reduce_function;
    
    int *main_port_connections = (int *) malloc(MAX_CELLS * sizeof(int));
    int *cell_types =(int *) malloc(MAX_CELLS * sizeof(int));
    for (int i = 0; i < MAX_CELLS; i++) {
      Cell *c = net[i];
      main_port_connections[i] = -1;
      cell_types[i] = -1;
      if (c == NULL) continue;
      // this way if net[c->ports[0]->connected_cell] main port should be = to i.
      main_port_connections[i] = c->ports[0].connected_cell;
      cell_types[i] = c->type;
    }

    int *cell_conns = (int *) malloc(MAX_CELLS * sizeof(int));
    int *conn_rules = (int *) malloc(MAX_CELLS * sizeof(int));

    find_reducible_c(main_port_connections, cell_types, cell_conns, conn_rules);

    // after the gpu find_reducible, we get an array with the connections and the already valid rules. Now we simply get conditionally apply the functions 
    for (int i = 0; i < MAX_CELLS; i++) {
      int conn = cell_conns[i];
      int rule = conn_rules[i];
      // no rule existent or already used
      if (conn == -1) continue;

      Cell *a = net[i];
      Cell *b = net[conn];

      if (rule == SUC_SUM) {
        if (a->type == SUM && b->type == SUC) {
          suc_sum(net, b, a);
        } else {
          suc_sum(net, a, b);
        }
      } else if (rule == ZERO_SUM) {
        if (a->type == SUM && b->type == ZERO) {
          zero_sum(net, b, a);
        } else {
          zero_sum(net, a, b);
        }
      }
      cell_conns[i] = -1;
    }

    // while(find_reducible(net, &reduce_function, &a_id, &b_id)) {
    //     if(net[a_id]->type == SUM && net[b_id]->type == SUC) {
    //         reduce_function(net, net[b_id], net[a_id]);
    //     } else if (net[a_id]->type == SUM && net[b_id]->type == ZERO) {
    //         reduce_function(net, net[b_id], net[a_id]);
    //     } else {
    //         reduce_function(net, net[a_id], net[b_id]);
    //     }
    // }

    // int val = church_decode(net);
    // printf("Decoded value: %d\n", val);
    
    for (int i = 0; i < MAX_CELLS; ++i) {
        if (net[i] != NULL) {
            delete_cell(net, i);
        }
    }
    free_ast(ast);

    return 0;   
}

