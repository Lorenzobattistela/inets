#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_AUX_PORTS 5
#define MAX_PORTS 10
#define MAX_CONNECTIONS 20

typedef struct Port Port;
typedef struct Cell Cell;
typedef struct Rule Rule;
typedef struct Net Net;
typedef void (*FunctionPtr)(Net *, Cell *, Cell *);

typedef enum {
  ERA, // erasor
  DUP,
  SUC,
  ZERO,
  SUM,
  MUL,
  ANY,
  NULL_S,
} Symbol;

void pprint_symbol(Symbol s);
Port *create_port(bool is_principal);
void print_port(Port *p);
Cell *create_cell(Symbol symbol, Port *principal_port, Port *auxiliary_ports[],
                  int aux_length);
void print_cell(Cell *cell);
void connect(Port *p, Port *q);
void erase(Net *net, Cell *erasor, Cell *to_erase);
void print_net(Net *net);
Cell *clone_cell(Cell *cell);
Cell *suc_cell(Net *net);

struct Port {
  bool is_principal;
  Port *connected_to;
  int connections;
};

struct Cell {
  Symbol symbol;
  Port *auxiliary_ports[MAX_AUX_PORTS];
  int aux_ports_length;
  Port *principal_port;
  bool deleted;
};

struct Rule {
  Symbol s1;
  Symbol s2;
  FunctionPtr reduce;
};

struct Net {
  Cell *cells[MAX_CONNECTIONS];
  int cell_count;
};

Cell *dup_cell(Net *net, int num_aux);
Rule dup_rule();

Cell *erasor_cell(Net *net);
Rule erase_rule();
void add_cell_to_net(Net *net, Cell *cell);
void delete_cell(Net *net, Cell *cell);

Cell *dup_cell(Net *net, int num_aux) {
  Port *principal_port = create_port(true);
  Cell *duplicator = create_cell(DUP, principal_port, NULL, 0);
  add_cell_to_net(net, duplicator);
  return duplicator;
}

void duplicate(Net *net, Cell *duplicator, Cell *to_duplicate) {
  if (duplicator->symbol != DUP || duplicator->deleted ||
      to_duplicate->deleted) {
    return;
  }
  // a duplicate rule does the following: we dup the to_dup cell,
  // connect the N aux ports of the to_dup cell with N ports (exchainging) of
  // the dup cells.
  // therefore we need to clone the to_dup_cell
  Cell *cloned = clone_cell(to_duplicate);
  Cell *dup_1 = dup_cell(net, to_duplicate->aux_ports_length);
  Cell *dup_2 = dup_cell(net, to_duplicate->aux_ports_length);

  for (int i = 0; i < to_duplicate->aux_ports_length; i++) {
    Port *aux_1 = create_port(false);
    Port *aux_2 = create_port(false);
    dup_1->auxiliary_ports[i] = aux_1;
    dup_2->auxiliary_ports[i] = aux_2;
  }

  // cloned cell and create the dup ones, now we can apply the rule
  // first dup cell aux port will connect with the first aux port of the dup
  // cells
}

Rule dup_rule() {
  Rule dup_r;
  dup_r.s1 = DUP;
  dup_r.s2 = NULL_S;
  dup_r.reduce = duplicate;
  return dup_r;
}

Cell *erasor_cell(Net *net) {
  Port *principal_port = create_port(true);
  Cell *erasor = create_cell(ERA, principal_port, NULL, 0);
  add_cell_to_net(net, erasor);
  return erasor;
}

void erase(Net *net, Cell *erasor, Cell *to_erase) {
  if (erasor->symbol != ERA || erasor->deleted || to_erase->deleted) {
    return;
  }
  for (int i = 0; i < to_erase->aux_ports_length; i++) {
    if (to_erase->auxiliary_ports[i]->connected_to) {
      Cell *e = erasor_cell(net);
      connect(to_erase->auxiliary_ports[i]->connected_to, e->principal_port);
    }
  }
  delete_cell(net, to_erase);
}

void suc_sum(Net *net, Cell *suc, Cell *sum) {
  if (suc->symbol != SUC || sum->symbol != SUM || suc->deleted ||
      sum->deleted) {
    return;
  }
  // a suc port connected with a + port
  // makes the second aux port connected to a suc cell intermediately
  // and connects the + main port to whatever the suc aux port was connected
  Cell *new_suc = suc_cell(net);

  // first connect sum main port to s aux port conn
  connect(sum->principal_port, suc->auxiliary_ports[0]->connected_to);
  // conn the new suc main port to sum aux connected to
  connect(new_suc->principal_port, sum->auxiliary_ports[1]->connected_to);
  // conn the new suc auxiliar to the + second aux
  connect(new_suc->auxiliary_ports[0], sum->auxiliary_ports[1]);

  // now simply delete the old suc
  delete_cell(net, suc);
}

Cell *sum_cell(Net *net) {
  // x
  Port *principal_port = create_port(true);
  Port *aux_ports[2];
  // y
  aux_ports[0] = create_port(false);
  // x + y
  aux_ports[1] = create_port(false);
  Cell *sum = create_cell(SUM, principal_port, aux_ports, 2);
  add_cell_to_net(net, sum);
  return sum;
}

Cell *suc_cell(Net *net) {
  Port *principal_port = create_port(true);
  Port *aux_ports[1];
  aux_ports[0] = create_port(false);
  Cell *suc = create_cell(SUC, principal_port, aux_ports, 1);
  add_cell_to_net(net, suc);
  return suc;
}

Rule erase_rule() {
  Rule erase_r;
  erase_r.s1 = ERA;
  erase_r.s2 = NULL_S;
  erase_r.reduce = erase;
  return erase_r;
}

Cell *clone_cell(Cell *cell) {
  if (cell == NULL) {
    return NULL;
  }

  Cell *c = (Cell *)malloc(sizeof(Cell));
  if (c == NULL) {
    return NULL;
  }

  c->symbol = cell->symbol;

  c->principal_port = (Port *)malloc(sizeof(Port));
  if (c->principal_port == NULL) {
    free(c);
    return NULL;
  }
  *c->principal_port = *(cell->principal_port);
  c->aux_ports_length = cell->aux_ports_length;
  for (int i = 0; i < cell->aux_ports_length; ++i) {
    c->auxiliary_ports[i] = (Port *)malloc(sizeof(Port));
    if (c->auxiliary_ports[i] == NULL) {
      for (int j = 0; j < i; ++j) {
        free(c->auxiliary_ports[j]);
      }
      free(c->principal_port);
      free(c);
      return NULL;
    }
    *c->auxiliary_ports[i] = *(cell->auxiliary_ports[i]);
  }

  c->deleted = cell->deleted;
  return c;
}

Cell *create_cell(Symbol symbol, Port *principal_port, Port *auxiliary_ports[],
                  int aux_length) {
  Cell *c = (Cell *)malloc(sizeof(Cell));
  c->symbol = symbol;
  c->principal_port = principal_port;
  c->aux_ports_length = aux_length;
  c->deleted = false;
  for (int i = 0; i < aux_length; i++) {
    c->auxiliary_ports[i] = auxiliary_ports[i];
  }
  return c;
}

void print_cell(Cell *cell) {
  pprint_symbol(cell->symbol);
  printf("Cell principal_port\n");
  print_port(cell->principal_port);
  printf("This cell has %i aux ports\n", cell->aux_ports_length);
  for (int i = 0; i < cell->aux_ports_length; i++) {
    Port *p = cell->auxiliary_ports[i];
    if (p != NULL) {
      print_port(p);
    }
  }
}

Port *create_port(bool is_principal) {
  Port *port = (Port *)malloc(sizeof(Port));
  port->is_principal = is_principal;
  port->connections = 0;
  port->connected_to = NULL;
  return port;
}

void print_port(Port *p) {
  if (p->is_principal) {
    printf("PRINCIPAL_PORT");
  } else {
    printf("AUXILIAR_PORT");
  }
  printf("\nConnected to %i ports\n", p->connections);
}

void print_net(Net *net) {
  for (int i = 0; i < net->cell_count; i++) {
    if (!net->cells[i]->deleted) {
      printf("Cell %d:\n", i);
      print_cell(net->cells[i]);
    }
  }
}

void connect(Port *p, Port *q) {
  if (p == NULL || q == NULL) {
    return;
  }
  p->connected_to = q;
  p->connections = 1;
  q->connected_to = p;
  q->connections = 1;
}

void pprint_symbol(Symbol s) {
  switch (s) {
  case ERA:
    printf("ERASER");
    break;
  case ANY:
    printf("ANY");
    break;
  default:
    return;
  }
  printf("\n");
}

bool reducible(Cell *a, Cell *b) { return false; }

void add_cell_to_net(Net *net, Cell *cell) {
  if (net->cell_count < MAX_CONNECTIONS) {
    net->cells[net->cell_count++] = cell;
  }
}

void delete_cell(Net *net, Cell *cell) {
  if (!cell->deleted) {
    cell->deleted = true;
    free(cell->principal_port);
    for (int i = 0; i < cell->aux_ports_length; i++) {
      free(cell->auxiliary_ports[i]);
    }
    free(cell);
  }
}

int main() {
  Net net;
  net.cell_count = 0;

  Port *main_port = create_port(true);
  Port *aux_ports[2];
  aux_ports[0] = create_port(false);
  aux_ports[1] = create_port(false);

  Cell *main_cell = create_cell(ANY, main_port, aux_ports, 2);
  add_cell_to_net(&net, main_cell);

  Port *other_port = create_port(true);
  Port *other_aux_ports[2];
  other_aux_ports[0] = create_port(false);
  other_aux_ports[1] = create_port(false);

  Cell *erasor = erasor_cell(&net);
  Cell *other_cell = create_cell(ANY, other_port, other_aux_ports, 2);
  add_cell_to_net(&net, other_cell);

  connect(main_cell->principal_port, erasor->principal_port);
  connect(main_cell->auxiliary_ports[0], other_cell->principal_port);

  Rule r = erase_rule();
  r.reduce(&net, erasor, main_cell);

  print_net(&net);

  for (int i = 0; i < net.cell_count; i++) {
    if (!net.cells[i]->deleted) {
      delete_cell(&net, net.cells[i]);
    }
  }

  return 0;
}
