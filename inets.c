#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "parser.h"

#define MAX_CELLS 35000
#define MAX_PORTS 3

#define SUM 0
#define SUC 1
#define ZERO 2

#define SUC_SUM (SUM + SUC)
#define ZERO_SUM (ZERO + SUM)

typedef struct {
    int connected_cell;
    int connected_port;
} Port;

typedef struct {
    int cell_id;
    int type;
    int num_aux_ports;
    Port *ports;
} Cell;

static int cell_counter = 0;
static int interactions = 0;

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

void delete_cell(int cell_id, int **arr_net, int **arr_ports, int *cell_types) {
    for (int i = 0; i < MAX_PORTS; i++) {
        arr_net[cell_id][i] = -1;
        arr_ports[cell_id][i] = -1;
    }
    cell_types[cell_id] = -1;
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

void suc_sum(int **arr_net, int **arr_ports, int *cell_types, int suc, int s) {
  int new_suc = suc_cell(arr_net, arr_ports, cell_types);

  int suc_first_aux_cell = arr_net[suc][1];
  int suc_first_aux_ports = arr_ports[suc][1];

  link(arr_net, arr_ports, cell_types, s, 0, suc_first_aux_cell, suc_first_aux_ports);
  link(arr_net, arr_ports, cell_types, new_suc, 1, arr_net[s][1], arr_ports[s][1]);
  link(arr_net, arr_ports, cell_types, new_suc, 0, s, 1);
  delete_cell(suc, arr_net, arr_ports, cell_types);
  interactions++;

}

void zero_sum(int **arr_net, int **arr_ports, int *cell_types, int zero, int s) {
  int sum_aux_first_connected_cell = arr_net[s][1];
  int sum_aux_first_connected_port = arr_ports[s][1];

  int sum_aux_snd_connected_cell = arr_net[s][2];
  int sum_aux_snd_connected_port = arr_ports[s][2];

  link(arr_net, arr_ports, cell_types, sum_aux_first_connected_cell, sum_aux_first_connected_port, sum_aux_snd_connected_cell, sum_aux_snd_connected_port);
  delete_cell(zero, arr_net, arr_ports, cell_types);
  delete_cell(s, arr_net, arr_ports, cell_types);
  interactions++;
}

void reduce(int *cell_conns, int *cell_types, int *conn_rules, int **arr_cell, int **arr_ports) {
    for (int i = 0; i < cell_counter; i++) {
        int conn = cell_conns[i];
        if (conn == -1) continue;

        int rule = conn_rules[i];

        int *a_connected_cell = arr_cell[i];
        int *a_connected_port = arr_ports[i];
        int a_type = cell_types[i];

        int *b_connected_cell = arr_cell[conn];
        int *b_connected_port = arr_ports[conn];
        int b_type = cell_types[conn];

        if (a_connected_cell == NULL || a_connected_port == NULL || b_connected_cell == NULL || b_connected_port == NULL || cell_types[i] == -1 || cell_types[conn] == -1) {
            continue;
        }

        if (rule == SUC_SUM) {
            if (a_type == SUM && b_type == SUC) {
              suc_sum(arr_cell, arr_ports, cell_types, conn, i);
            } else {
              suc_sum(arr_cell, arr_ports, cell_types, i, conn);
            }
        } else if (rule == ZERO_SUM) {
            if (a_type == SUM && b_type == ZERO) {
              zero_sum(arr_cell, arr_ports, cell_types, conn, i);
            } else {
              zero_sum(arr_cell, arr_ports, cell_types, i, conn);
            }
        }
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

bool is_valid_rule (int rule) {
    return (rule == SUC_SUM) || (rule == ZERO_SUM);
}

int find_reducible(int **arr_cells, int **arr_ports, int *cell_conns, int *cell_types, int *conn_rules) {
    int found = 0;
    for (int i = 0; i < cell_counter; i++) {
        cell_conns[i] = -1;
        conn_rules[i] = -1;

        int *main_port = arr_cells[i];
        if (main_port == NULL) {
            continue;
        }
        int main_port_conn = main_port[0];
        if (main_port_conn == -1) {
            continue;
        }

        int *connection_port = arr_cells[main_port_conn];
        if (connection_port == NULL) {
            continue;
        }
        int connection_main = connection_port[0];
        
        if (cell_types[i] == -1 || cell_types[main_port_conn] == -1) {
            continue;
        }

        int rule = cell_types[i] + cell_types[main_port_conn];

        if(!is_valid_rule(rule)) {
            continue;
        }

        if (connection_main != i) {
            continue;
        }

        if (i > main_port_conn) {
            continue;
        }
        cell_conns[i] = main_port_conn;
        conn_rules[i] = rule;
        found++;
    }
    return found;
}

int main() {
    const char *in = "(((1 + 10) + (9 + 1)) + ((1 + 1) + (1 + 1))) + (((1 + 1) + (1 + 1)) + ((1 + 1) + (1 + 1))) + 1" ;

    ASTNode *ast = parse(in);

    int **arr_cells = (int **) malloc(MAX_CELLS * sizeof(int *));
    int **arr_ports = (int **) malloc(MAX_CELLS * sizeof(int *));
    int *cell_types =(int *) malloc(MAX_CELLS * sizeof(int));
    int *cell_conns =(int *) malloc(MAX_CELLS * sizeof(int));
    int *conn_rules =(int *) malloc(MAX_CELLS * sizeof(int));
  
    for (int i = 0; i < MAX_CELLS; i++) {
      arr_cells[i] = (int *)malloc(MAX_PORTS * sizeof(int));
      arr_ports[i] = (int *)malloc(MAX_PORTS * sizeof(int));
      cell_types[i] = -1;
      cell_conns[i] = -1;
      conn_rules[i] = -1;
      for (int j = 0; j < MAX_PORTS; j++) {
        arr_cells[i][j] = -1;
        arr_ports[i][j] = -1;
      }
    }
    // print_ast(ast);
    to_interaction_net(ast, arr_cells, arr_ports, cell_types);
    interactions = 0;

    clock_t start, end;
    double cpu_time_used;

    int reductions = find_reducible(arr_cells, arr_ports, cell_conns, cell_types, conn_rules);
    start = clock();
    
    while (reductions > 0) {
        reduce(cell_conns, cell_types, conn_rules, arr_cells, arr_ports);
        reductions = find_reducible(arr_cells, arr_ports, cell_conns, cell_types, conn_rules);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    double ips = (double)interactions / cpu_time_used;

    int val = church_decode(arr_cells, arr_ports, cell_types);
    printf("Decoded value: %d\n", val);

    printf("The program took %f seconds to execute and made %i interactions.\n", cpu_time_used, interactions);
    printf("Interactions per second: %f\n", ips);
    
    free_ast(ast);
    for (int i = 0; i < MAX_CELLS; i++) {
      free(arr_cells[i]);
      free(arr_ports[i]);
    }
    free(cell_conns);
    free(conn_rules);
    free(cell_types);
    free(arr_cells);
    free(arr_ports);
    return 0;   
}
