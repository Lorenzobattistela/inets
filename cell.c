#include "cell.h"


bool check_mul_zero(Connection *c) {









Cell *dup_cell(Net *net) {
  Port *principal_port = create_port(true);
  Port *aux_ports[2];
  aux_ports[0] = create_port(false);
  aux_ports[1] = create_port(false);
  Cell *dup = create_cell(SUM, principal_port, aux_ports, 2);
  add_cell_to_net(net, dup);
  return dup;
}

Cell *mul_cell(Net *net) {
  Port *principal_port = create_port(true);
  Port *aux_ports[2];
  aux_ports[0] = create_port(false);
  aux_ports[1] = create_port(false);
  Cell *dup = create_cell(MUL, principal_port, aux_ports, 2);
  add_cell_to_net(net, dup);
  return dup;
}

void mul_suc(Net *net, Cell *mul, Cell *suc) {
  if (mul->symbol != MUL || suc->symbol != SUC || mul->deleted ||
      suc->deleted) {
    return;
  }
  // this rule is a little more complex. We will end up with a mul port, a sum
  // port and a dup port
  // first, we connect mul princ port to suc aux port conn
  // then we create a sum and a dup port
  // connect the sum princ port to the 2 aux port of mul
  // connect dup main port to whatever mul 1st aux port was connected to
  // connect the dup 1 aux port to 1 aux port of mul
  // connected the dup 2 aux port to 1 aux port of sum
  // delete suc
  connect(mul->principal_port, suc->auxiliary_ports[0]->connected_to);
  Cell *sum = sum_cell(net);
  Cell *dup = dup_cell(net);

  connect(sum->auxiliary_ports[1], mul->auxiliary_ports[1]->connected_to);
  connect(sum->principal_port, mul->auxiliary_ports[1]);
  connect(dup->principal_port, mul->auxiliary_ports[0]->connected_to);
  connect(mul->auxiliary_ports[0], dup->auxiliary_ports[0]);
  connect(dup->auxiliary_ports[1], sum->auxiliary_ports[0]);

  delete_cell(net, suc);
}

void mul_zero(Net *net, Cell *mul, Cell *zero) {
  if (mul->symbol != MUL || zero->symbol != ZERO || mul->deleted ||
      zero->deleted) {
    return;
  }
  // mul and zero interaction basically connects the first
  // aux port of mul to an erasor and the second one to zero
  Cell *erasor = erasor_cell(net);
  connect(mul->auxiliary_ports[0]->connected_to, erasor->principal_port);
  connect(mul->auxiliary_ports[1]->connected_to, zero->principal_port);
  delete_cell(net, mul);
  return;
}

void zero_erasor(Net *net, Cell *zero, Cell *erasor) {
  if (zero->symbol != SUC || erasor->symbol != ERA || zero->deleted ||
      erasor->deleted) {
    return;
  }
  // simply delete both cells, nothing remains
  delete_cell(net, zero);
  delete_cell(net, erasor);
}

void suc_erasor(Net *net, Cell *suc, Cell *erasor) {
  if (suc->symbol != SUC || erasor->symbol != ERA || suc->deleted ||
      erasor->deleted) {
    return;
  }

  // erase a suc simply deletes the suc cell and connects the erasor to whatever
  // suc aux port was connected to
  connect(erasor->principal_port, suc->auxiliary_ports[0]->connected_to);
  delete_cell(net, suc);
}

void zero_dup(Net *net, Cell *zero, Cell *dup) {
  if (zero->symbol != ZERO || dup->symbol != DUP || zero->deleted ||
      dup->deleted) {
    return;
  }
  // zero main to dup main should clone 0 and connect it both
  // to whatever was connected to dup aux nodes
  Cell *new_zero = zero_cell(net);
  connect(zero->principal_port, dup->auxiliary_ports[0]->connected_to);
  connect(new_zero->principal_port, dup->auxiliary_ports[1]->connected_to);
  delete_cell(net, dup);
  return;
}

void suc_dup(Net *net, Cell *suc, Cell *dup) {
  if (suc->symbol != SUC || dup->symbol != DUP || suc->deleted ||
      dup->deleted) {
    return;
  }
  // if a suc main port connects to a dup main port, simply return
  // two sucs with main ports connected to the aux ports of dup
  // therefore we create a new suc and connect the old one
  Cell *new_suc = suc_cell(net);
  // connect dup principal port with suc aux conn
  connect(dup->principal_port, suc->auxiliary_ports[0]->connected_to);
  // conn suc principal_port to dup aux conn
  connect(suc->principal_port, dup->auxiliary_ports[0]->connected_to);
  connect(new_suc->principal_port, dup->auxiliary_ports[1]->connected_to);

  // connect suc and new suc aux ports with dup aux ports
  connect(suc->auxiliary_ports[0], dup->auxiliary_ports[0]);
  connect(new_suc->auxiliary_ports[0], dup->auxiliary_ports[1]);
  return;
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

void zero_sum(Net *net, Cell *zero, Cell *sum) {
  if (zero->symbol != ZERO || sum->symbol != SUM || zero->deleted ||
      sum->deleted) {
    return;
  }
  // for this rule, we simply connect whatever is connected to 1st aux
  // port of the sum rule to whatever in the 2nd aux
  connect(sum->auxiliary_ports[0]->connected_to,
          sum->auxiliary_ports[1]->connected_to);
  // and delete both + and 0
  delete_cell(net, zero);
  delete_cell(net, sum);
}

Cell *zero_cell(Net *net) {
  Port *principal_port = create_port(0);
  Cell *zero = create_cell(ZERO, principal_port, NULL, 0);
  add_cell_to_net(net, zero);
  return zero;
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
