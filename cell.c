#include "cell.h"
#include <stdio.h>

bool check_mul_suc(Connection *c) {
  return (c->a->symbol == MUL && c->b->symbol == SUC) &&
         (c->a->symbol == SUC && c->b->symbol == MUL);
}

bool check_mul_zero(Connection *c) {
  return (c->a->symbol == MUL && c->b->symbol == ZERO) ||
         (c->a->symbol == ZERO && c->b->symbol == MUL);
}

bool check_zero_erasor(Connection *c) {
  return (c->a->symbol == ZERO && c->b->symbol == ERA) ||
         (c->a->symbol == ERA && c->b->symbol == ZERO);
}

bool check_suc_erasor(Connection *c) {
  return (c->a->symbol == SUC && c->b->symbol == ERA) ||
         (c->a->symbol == ERA && c->b->symbol == SUC);
}

bool check_zero_dup(Connection *c) {
  return (c->a->symbol == ZERO && c->b->symbol == DUP) ||
         (c->a->symbol == DUP && c->b->symbol == ZERO);
}

bool check_suc_dup(Connection *c) {
  return (c->a->symbol == SUC && c->b->symbol == DUP) ||
         (c->a->symbol == DUP && c->b->symbol == SUC);
}

bool check_suc_sum(Connection *c) {
  return (c->a->symbol == SUC && c->b->symbol == SUM) ||
         (c->a->symbol == SUM && c->b->symbol == SUC);
}

bool check_zero_sum(Connection *c) {
  return (c->a->symbol == ZERO && c->b->symbol == SUM) ||
         (c->a->symbol == SUM && c->b->symbol == ZERO);
}

Rule mul_suc_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = mul_suc;
  rule.reducible = true;
  return rule;
}

Rule mul_zero_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = mul_zero;
  rule.reducible = true;
  return rule;
}

Rule zero_erasor_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = zero_erasor;
  rule.reducible = true;
  return rule;
}

Rule suc_erasor_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = suc_erasor;
  rule.reducible = true;
  return rule;
}

Rule zero_dup_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = zero_dup;
  rule.reducible = true;
  return rule;
}

Rule suc_dup_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = suc_dup;
  rule.reducible = true;
  return rule;
}

Rule suc_sum_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = suc_sum;
  rule.reducible = true;
  return rule;
}

Rule zero_sum_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = zero_sum;
  rule.reducible = true;
  return rule;
}

Rule do_nothing_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = do_nothing;
  rule.reducible = false;
  return rule;
}

void do_nothing(Net *net, Cell *a, Cell *b, Connection *conn) { return; }

Rule match_with_rule(Connection *c) {
  if (check_mul_suc(c)) {
    return mul_suc_rule(c);
  } else if (check_mul_zero(c)) {
    return mul_zero_rule(c);
  } else if (check_zero_erasor(c)) {
    return zero_erasor_rule(c);
  } else if (check_suc_erasor(c)) {
    return suc_erasor_rule(c);
  } else if (check_zero_dup(c)) {
    return zero_dup_rule(c);
  } else if (check_suc_dup(c)) {
    return suc_dup_rule(c);
  } else if (check_suc_sum(c)) {
    return suc_sum_rule(c);
  } else if (check_zero_sum(c)) {
    return zero_sum_rule(c);
  }
  return do_nothing_rule(c);
}

void update_connections(Net *net) {
  for (int i = 0; i < net->cell_count; ++i) {
    Cell *cell_a = net->cells[i];
    if (cell_a->deleted || cell_a->principal_port == NULL ||
        !cell_a->principal_port->is_principal) {
      continue;
    }

    Port *port_a = cell_a->principal_port->connected_to;
    if (port_a == NULL || !port_a->is_principal) {
      continue;
    }

    for (int j = 0; j < net->cell_count; ++j) {
      if (i == j)
        continue;
      Cell *cell_b = net->cells[j];
      if (cell_b->deleted || cell_b->principal_port == NULL) {
        continue;
      }

      if (cell_b->principal_port == port_a) {
        bool exists = false;
        for (int k = 0; k < net->connection_count; ++k) {
          Connection *conn = net->connections[k];
          if ((conn->a == cell_a && conn->b == cell_b) ||
              (conn->a == cell_b && conn->b == cell_a)) {
            exists = true;
            break;
          }
        }

        if (!exists && net->connection_count < MAX_CONNECTIONS) {
          Connection *new_conn = (Connection *)malloc(sizeof(Connection));
          if (new_conn != NULL) {
            new_conn->a = cell_a;
            new_conn->b = cell_b;
            new_conn->used = false;
            net->connections[net->connection_count++] = new_conn;
          } else {
          }
        }
        break;
      }
    }
  }
}

// this should probably be parallelizable
Rule find_reducible(Net *net) {
  for (int i = 0; i < net->connection_count; i++) {
    Connection *conn = net->connections[i];
    if (conn->used) {
      continue;
    }
    Rule r = match_with_rule(conn);
    if (r.reducible) {
      return r;
    }
  }
  Rule null;
  null.c = NULL;
  null.reduce = do_nothing;
  null.reducible = false;
  return null;
}

// ============== Specific Cell creation functions ================
Cell *suc_cell(Net *net) {
  Port *principal_port = create_port(true);
  Port *aux_ports[1];
  aux_ports[0] = create_port(false);
  Cell *suc = create_cell(SUC, principal_port, aux_ports, 1);
  add_cell_to_net(net, suc);
  return suc;
}

Cell *zero_cell(Net *net) {
  Port *principal_port = create_port(true);
  Cell *zero = create_cell(ZERO, principal_port, NULL, 0);
  add_cell_to_net(net, zero);
  return zero;
}

Cell *dup_cell(Net *net) {
  Port *principal_port = create_port(true);
  Port *aux_ports[2];
  aux_ports[0] = create_port(false);
  aux_ports[1] = create_port(false);
  Cell *dup = create_cell(SUM, principal_port, aux_ports, 2);
  add_cell_to_net(net, dup);
  return dup;
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

Cell *erasor_cell(Net *net) {
  Port *principal_port = create_port(true);
  Cell *erasor = create_cell(ERA, principal_port, NULL, 0);
  add_cell_to_net(net, erasor);
  return erasor;
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

// ================ Cell interaction functions ===================
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

Port *create_port(bool is_principal) {
  Port *port = (Port *)malloc(sizeof(Port));
  port->is_principal = is_principal;
  port->connections = 0;
  port->connected_to = NULL;
  return port;
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

Net create_net() {
  Net n;
  n.connection_count = 0;
  n.cell_count = 0;
  return n;
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
  }
}

// ========================= Pretty printing functions ==============
void print_net(Net *net) {
  for (int i = 0; i < net->cell_count; i++) {
    if (!net->cells[i]->deleted) {
      printf("Cell %d:\n", i);
      print_cell(net->cells[i]);
    }
  }
}

void print_port(Port *p) {
  if (p->is_principal) {
    printf("PRINCIPAL_PORT");
  } else {
    printf("AUXILIAR_PORT");
  }
  printf("\nConnected to %i ports\n", p->connections);
}

void pprint_symbol(Symbol s) {
  switch (s) {
  case ERA:
    printf("ERASER");
    break;
  case DUP:
    printf("DUPLICATOR");
    break;
  case SUC:
    printf("SUCCESSOR");
    break;
  case ZERO:
    printf("ZERO");
    break;
  case SUM:
    printf("SUM");
    break;
  case MUL:
    printf("MUL");
    break;
  case ANY:
    printf("ANY");
    break;
  default:
    return;
  }
  printf("\n");
}

void print_cell(Cell *cell) {
  if (cell->deleted) {
    printf("DELETED CELL!\n");
    return;
  }
  printf("Symbol: ");
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

void print_rule(Rule *rule) {
  printf("Rule connection:\n");
  pprint_symbol(rule->c->a->symbol);
  pprint_symbol(rule->c->b->symbol);
}

void print_connection(Connection *connection) {
  printf("Connection:");
  printf("A cell:");
  print_cell(connection->a);
  printf("B cell:");
  print_cell(connection->b);
}

int main() {
  Net net = create_net();
  Cell *z = zero_cell(&net);
  Cell *s = suc_cell(&net);
  // suc(0)
  connect(z->principal_port, s->auxiliary_ports[0]);

  Cell *z_1 = zero_cell(&net);
  Cell *sum_c = sum_cell(&net);
  // y = 0
  connect(z_1->principal_port, sum_c->auxiliary_ports[0]);
  // x = suc(0)
  connect(s->principal_port, sum_c->principal_port);
  Connection conn;
  conn.a = s;
  conn.b = sum_c;
  conn.used = false;
  net.connections[net.connection_count] = &conn;
  net.connection_count++;
  // print_net(&net);
  Rule r = find_reducible(&net);
  // print_rule(&r);
  r.reduce(&net, r.c->a, r.c->b, &conn);
  // print_net(&net);
  update_connections(&net);
  for (int j = 0; j < net.connection_count; j++) {
    Connection *c = net.connections[j];
    printf("IS C NULL? %i\n", c == NULL);
    if (!c->used) {
      print_connection(c);
    }
  }
  Rule r2 = find_reducible(&net);
  print_rule(&r2);
  r2.reduce(&net, r2.c->a, r2.c->b, r2.c);
  print_net(&net);
}
