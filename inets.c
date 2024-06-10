#include "memory.h"
#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_CELLS 35000
#define MAX_PORTS 3

#define SUM (1 << 0)  // 1
#define SUC (1 << 1)  // 2
#define ZERO (1 << 2) // 4
#define MUL (1 << 3)  // 8
#define DUP (1 << 4)  // 16
#define ERA (1 << 5)  // 32

#define SUC_SUM (SUC | SUM)   // 3
#define ZERO_SUM (ZERO | SUM) // 5
#define SUC_MUL (SUC | MUL)   // 10
#define ZERO_MUL (ZERO | MUL) // 12
#define SUC_DUP (SUC | DUP)   // 18
#define ZERO_DUP (ZERO | DUP) // 20
#define SUC_ERA (SUC | ERA)   // 34
#define ZERO_ERA (ZERO | ERA) // 36

typedef struct {
  int cell_id;
  int connected_cell_id;
} Redex;

typedef struct {
  int count;
  int capacity;
  Redex **entries;
} Redexes;

typedef void (*interaction)(int **, int **, int *, Redexes *, int, int);

void init_redexes(Redexes *redexes) {
  redexes->count = 0;
  redexes->capacity = 0;
  redexes->entries = NULL;
}

void init_redexes_with_capacity(Redexes *redexes, int capacity) {
  redexes->count = 0;
  redexes->capacity = capacity;
  redexes->entries = ALLOCATE(Redex *, capacity);
}

void write_redex(Redexes *redexes, Redex *redex) {
  if (redexes->capacity < redexes->count + 1) {
    int oldCapacity = redexes->capacity;
    redexes->capacity = GROW_CAPACITY(oldCapacity);
    redexes->entries =
        GROW_ARRAY(Redex *, redexes->entries, oldCapacity, redexes->capacity);
  }
  redexes->entries[redexes->count] = redex;
  redexes->count++;
}

void free_redexes(Redexes *redexes) {
  FREE_ARRAY(Redex, redexes, redexes->capacity);
  init_redexes(redexes);
}

void print_redexes(Redexes *redexes) {
  for (int i = 0; i < redexes->count; i++) {
    printf("Redex at index %i\n Cell %i redex with cell %i\n", i,
           redexes->entries[i]->cell_id,
           redexes->entries[i]->connected_cell_id);
  }
}

Redex *pop(Redexes *redexes) {
  Redex *r = redexes->entries[redexes->count - 1];
  redexes->count--;
  return r;
}

void push(Redexes *redex, Redex *r) { write_redex(redex, r); }

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

int create_cell(int **arr_net, int **arr_ports, int *cell_types,
                int cell_type) {
  int cell_id = cell_counter++;
  cell_types[cell_id] = cell_type;
  for (int i = 0; i < MAX_PORTS; i++) {
    arr_net[cell_id][i] = -1;
    arr_ports[cell_id][i] = -1;
  }
  return cell_id;
}

int sum_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, SUM);
  return cell_id;
}

int suc_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, SUC);
  return cell_id;
}

int zero_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, ZERO);
  return cell_id;
}

int mul_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, MUL);
  return cell_id;
}

int dup_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, DUP);
  return cell_id;
}

int era_cell(int **arr_net, int **arr_ports, int *cell_types) {
  int cell_id = create_cell(arr_net, arr_ports, cell_types, ERA);
  return cell_id;
}

void link(int **arr_net, int **arr_ports, Redexes *redexes, int *cell_types,
          int a_id, int a_port, int b_id, int b_port) {
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

    if (a_port == 0 && arr_ports[a_id][a_port] == 0) {
      Redex *r = malloc(sizeof(Redex));
      if (r == NULL)
        exit(EXIT_FAILURE);

      r->cell_id = a_id;
      r->connected_cell_id = arr_net[a_id][a_port];
      write_redex(redexes, r);
    }
  }
}

void suc_sum(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
             int suc, int s) {
  int new_suc = suc_cell(arr_net, arr_ports, cell_types);

  int suc_first_aux_cell = arr_net[suc][1];
  int suc_first_aux_ports = arr_ports[suc][1];

  link(arr_net, arr_ports, redexes, cell_types, s, 0, suc_first_aux_cell,
       suc_first_aux_ports);
  link(arr_net, arr_ports, redexes, cell_types, new_suc, 1, arr_net[s][1],
       arr_ports[s][1]);
  link(arr_net, arr_ports, redexes, cell_types, new_suc, 0, s, 1);
  delete_cell(suc, arr_net, arr_ports, cell_types);
  interactions++;
}

void zero_sum(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
              int zero, int s) {
  int sum_aux_first_connected_cell = arr_net[s][1];
  int sum_aux_first_connected_port = arr_ports[s][1];

  int sum_aux_snd_connected_cell = arr_net[s][2];
  int sum_aux_snd_connected_port = arr_ports[s][2];

  link(arr_net, arr_ports, redexes, cell_types, sum_aux_first_connected_cell,
       sum_aux_first_connected_port, sum_aux_snd_connected_cell,
       sum_aux_snd_connected_port);
  delete_cell(zero, arr_net, arr_ports, cell_types);
  delete_cell(s, arr_net, arr_ports, cell_types);
  interactions++;
}

void suc_mul(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
             int suc, int mul) {

  int sum_c = sum_cell(arr_net, arr_ports, cell_types);
  int dup = dup_cell(arr_net, arr_ports, cell_types);

  // link main port of mul to what was connected to suc aux
  link(arr_net, arr_ports, redexes, cell_types, mul, 0, arr_net[suc][1],
       arr_ports[suc][1]);

  // link main port of dup to what was connected to mul 1st aux
  link(arr_net, arr_ports, redexes, cell_types, dup, 0, arr_net[mul][1],
       arr_ports[mul][1]);

  // link mul 1st aux to dup 2nd aux
  link(arr_net, arr_ports, redexes, cell_types, mul, 1, dup, 2);

  // link sum 2nd aux to what was connected to mul 2nd aux
  link(arr_net, arr_ports, redexes, cell_types, sum_c, 2, arr_net[mul][2],
       arr_ports[mul][2]);

  // link  sum main port to mul 2nd aux
  link(arr_net, arr_ports, redexes, cell_types, sum_c, 0, mul, 2);

  // link sum 1st aux to dup 1st aux
  link(arr_net, arr_ports, redexes, cell_types, sum_c, 1, dup, 1);

  delete_cell(suc, arr_net, arr_ports, cell_types);
}

void zero_mul(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
              int zero, int mul) {
  int era = era_cell(arr_net, arr_ports, cell_types);

  // link era main port to mul 1st aux
  link(arr_net, arr_ports, redexes, cell_types, era, 0, mul, 1);

  // link zero main port to mul snd aux
  link(arr_net, arr_ports, redexes, cell_types, zero, 0, mul, 2);

  delete_cell(mul, arr_net, arr_ports, cell_types);
}

void suc_dup(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
             int suc, int dup) {

  int new_suc = suc_cell(arr_net, arr_ports, cell_types);

  // link dup main to what was connected to suc aux
  link(arr_net, arr_ports, redexes, cell_types, dup, 0, arr_net[suc][1],
       arr_ports[suc][1]);

  // link suc main to what was connected to dup first aux
  link(arr_net, arr_ports, redexes, cell_types, suc, 0, arr_net[dup][1],
       arr_ports[dup][1]);

  // link suc 1st aux to dup 1st aux
  link(arr_net, arr_ports, redexes, cell_types, suc, 1, dup, 1);

  // link new suc main to what was connected to dup snd aux
  link(arr_net, arr_ports, redexes, cell_types, new_suc, 0, arr_net[dup][2],
       arr_ports[dup][2]);

  // link new suc first aux to dup 2nd aux
  link(arr_net, arr_ports, redexes, cell_types, suc, 1, dup, 2);
}

void zero_dup(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
              int zero, int dup) {
  int new_zero = zero_cell(arr_net, arr_ports, cell_types);

  // connect zero main port to what was connected on dup 1st aux
  link(arr_net, arr_ports, redexes, cell_types, zero, 0, arr_net[dup][1],
       arr_ports[dup][1]);

  // link new zero main port to what was connected on dup 2nd aux
  link(arr_net, arr_ports, redexes, cell_types, zero, 0, arr_net[dup][2],
       arr_ports[dup][2]);

  delete_cell(dup, arr_net, arr_ports, cell_types);
}

void suc_era(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
             int suc, int era) {
  // link era main to what was connected to suc first aux
  link(arr_net, arr_ports, redexes, cell_types, era, 0, arr_net[suc][1],
       arr_ports[suc][1]);

  delete_cell(suc, arr_net, arr_ports, cell_types);
}

void zero_era(int **arr_net, int **arr_ports, int *cell_types, Redexes *redexes,
              int zero, int era) {
  // deletes both
  delete_cell(zero, arr_net, arr_ports, cell_types);
  delete_cell(era, arr_net, arr_ports, cell_types);
}

bool should_exchange(int a, int b) { return a > b; }

interaction get_interaction(int rule) {
  switch (rule) {
  case SUC_SUM:
    return suc_sum;
  case ZERO_SUM:
    return zero_sum;
  default:
    return NULL;
  }
}

void reduce(int *cell_types, int **arr_cell, int **arr_ports,
            Redexes *redexes) {
  while (redexes->count > 0) {
    Redex *r = pop(redexes);

    if (r == NULL) {
      exit(EXIT_FAILURE);
      break;
    }

    int cell_id = r->cell_id;
    int conn_cell_id = r->connected_cell_id;

    int *a_connected_cell = arr_cell[cell_id];
    int *a_connected_port = arr_ports[cell_id];
    int a_type = cell_types[cell_id];

    int *b_connected_cell = arr_cell[conn_cell_id];
    int *b_connected_port = arr_ports[conn_cell_id];
    int b_type = cell_types[conn_cell_id];

    if (a_connected_cell == NULL || a_connected_port == NULL ||
        b_connected_cell == NULL || b_connected_port == NULL ||
        cell_types[cell_id] == -1 || cell_types[conn_cell_id] == -1) {
      continue;
    }

    int rule = a_type + b_type;
    interaction itr = get_interaction(rule);

    if (should_exchange(a_type, b_type)) {
      itr(arr_cell, arr_ports, cell_types, redexes, cell_id, conn_cell_id);
    } else {
      itr(arr_cell, arr_ports, cell_types, redexes, conn_cell_id, cell_id);
    }
  }
}

int church_encode(int **arr_cells, int **arr_ports, int *cell_types, int num) {
  int zero = zero_cell(arr_cells, arr_ports, cell_types);
  int to_connect_cell = zero;
  int to_connect_port = 0;

  for (int i = 0; i < num; i++) {
    int suc = suc_cell(arr_cells, arr_ports, cell_types);
    link(arr_cells, arr_ports, NULL, cell_types, suc, 1, to_connect_cell,
         to_connect_port);
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

int to_interaction_net(ASTNode *node, int **arr_cells, int **arr_ports,
                       int *cell_types, Redexes *redexes) {
  if (node == NULL)
    return -1;

  if (node->token == DIGIT) {
    return church_encode(arr_cells, arr_ports, cell_types, node->value);
  } else if (node->token == PLUS) {
    int left_cell_id = to_interaction_net(node->left, arr_cells, arr_ports,
                                          cell_types, redexes);
    int left_port = 0;
    int right_cell_id = to_interaction_net(node->right, arr_cells, arr_ports,
                                           cell_types, redexes);
    int right_port = 0;

    if (cell_types[left_cell_id] == SUM) {
      left_port = 2;
    }
    if (cell_types[right_cell_id] == SUM) {
      right_port = 2;
    }

    int sum = sum_cell(arr_cells, arr_ports, cell_types);
    // linking here
    link(arr_cells, arr_ports, redexes, cell_types, sum, 0, left_cell_id,
         left_port);
    link(arr_cells, arr_ports, redexes, cell_types, sum, 1, right_cell_id,
         right_port);
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
  const char *in = "10 + 10 + 10 + 10";
  ASTNode *ast = parse(in);

  int **arr_cells = (int **)malloc(MAX_CELLS * sizeof(int *));
  int **arr_ports = (int **)malloc(MAX_CELLS * sizeof(int *));
  int *cell_types = (int *)malloc(MAX_CELLS * sizeof(int));
  Redexes *redexes = malloc(sizeof(Redexes));
  init_redexes_with_capacity(redexes, 16);

  for (int i = 0; i < MAX_CELLS; i++) {
    arr_cells[i] = (int *)malloc(MAX_PORTS * sizeof(int));
    arr_ports[i] = (int *)malloc(MAX_PORTS * sizeof(int));
    cell_types[i] = -1;
    for (int j = 0; j < MAX_PORTS; j++) {
      arr_cells[i][j] = -1;
      arr_ports[i][j] = -1;
    }
  }
  // print_ast(ast);
  to_interaction_net(ast, arr_cells, arr_ports, cell_types, redexes);
  interactions = 0;

  clock_t start, end;
  double cpu_time_used;

  start = clock();

  while (redexes->count > 0) {
    reduce(cell_types, arr_cells, arr_ports, redexes);
  }
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  double ips = (double)interactions / cpu_time_used;

  int val = church_decode(arr_cells, arr_ports, cell_types);
  printf("Decoded value: %d\n", val);

  printf("The program took %f seconds to execute and made %i interactions.\n",
         cpu_time_used, interactions);
  printf("Interactions per second: %f\n", ips);

  free_ast(ast);
  for (int i = 0; i < MAX_CELLS; i++) {
    free(arr_cells[i]);
    free(arr_ports[i]);
  }
  free(cell_types);
  free(arr_cells);
  free(arr_ports);
  for (int j = 0; j < redexes->capacity; j++) {
    free(redexes->entries[j]);
  }
  free(redexes);
  return 0;
}
