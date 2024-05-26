#include "cell.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// all checks, connect, reduce rules and delete cell should be device

__device__ bool check_mul_suc(Connection *c) {
  return (c->a->symbol == MUL && c->b->symbol == SUC) &&
         (c->a->symbol == SUC && c->b->symbol == MUL);
}

__device__ bool check_mul_zero(Connection *c) {
  return (c->a->symbol == MUL && c->b->symbol == ZERO) ||
         (c->a->symbol == ZERO && c->b->symbol == MUL);
}

__device__ bool check_zero_erasor(Connection *c) {
  return (c->a->symbol == ZERO && c->b->symbol == ERA) ||
         (c->a->symbol == ERA && c->b->symbol == ZERO);
}

__device__ bool check_suc_erasor(Connection *c) {
  return (c->a->symbol == SUC && c->b->symbol == ERA) ||
         (c->a->symbol == ERA && c->b->symbol == SUC);
}

__device__ bool check_zero_dup(Connection *c) {
  return (c->a->symbol == ZERO && c->b->symbol == DUP) ||
         (c->a->symbol == DUP && c->b->symbol == ZERO);
}

__device__ bool check_suc_dup(Connection *c) {
  return (c->a->symbol == SUC && c->b->symbol == DUP) ||
         (c->a->symbol == DUP && c->b->symbol == SUC);
}

__device__ bool check_suc_sum(Connection *c) {
  return (c->a->symbol == SUC && c->b->symbol == SUM) ||
         (c->a->symbol == SUM && c->b->symbol == SUC);
}

__device__ bool check_zero_sum(Connection *c) {
  return (c->a->symbol == ZERO && c->b->symbol == SUM) ||
         (c->a->symbol == SUM && c->b->symbol == ZERO);
}

__device__ void mul_suc(Net *net, Cell *mul, Cell *suc, Connection *conn) {
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
  connect_c(mul->principal_port, suc->auxiliary_ports[0]->connected_to);
  Cell *sum = sum_cell_c(net);
  Cell *dup = dup_cell_c(net);

  connect_c(sum->auxiliary_ports[1], mul->auxiliary_ports[1]->connected_to);
  connect_c(sum->principal_port, mul->auxiliary_ports[1]);
  connect_c(dup->principal_port, mul->auxiliary_ports[0]->connected_to);
  connect_c(mul->auxiliary_ports[0], dup->auxiliary_ports[0]);
  connect_c(dup->auxiliary_ports[1], sum->auxiliary_ports[0]);

  delete_cell_c(net, suc);

  conn->used = true;
}

__device__ void mul_zero(Net *net, Cell *mul, Cell *zero, Connection *conn) {
  if (mul->symbol != MUL || zero->symbol != ZERO || mul->deleted ||
      zero->deleted) {
    return;
  }
  // mul and zero interaction basically connects the first
  // aux port of mul to an erasor and the second one to zero
  Cell *erasor = erasor_cell_c(net);
  connect_c(mul->auxiliary_ports[0]->connected_to, erasor->principal_port);
  connect_c(mul->auxiliary_ports[1]->connected_to, zero->principal_port);
  delete_cell_c(net, mul);

  conn->used = true;
  return;
}

__device__ void zero_erasor(Net *net, Cell *zero, Cell *erasor, Connection *conn) {
  if (zero->symbol != SUC || erasor->symbol != ERA || zero->deleted ||
      erasor->deleted) {
    return;
  }
  // simply delete both cells, nothing remains
  delete_cell_c(net, zero);
  delete_cell_c(net, erasor);

  conn->used = true;
}

__device__ void suc_erasor(Net *net, Cell *suc, Cell *erasor, Connection *conn) {
  if (suc->symbol != SUC || erasor->symbol != ERA || suc->deleted ||
      erasor->deleted) {
    return;
  }

  // erase a suc simply deletes the suc cell and connects the erasor to whatever
  // suc aux port was connected to
  connect_c(erasor->principal_port, suc->auxiliary_ports[0]->connected_to);
  delete_cell_c(net, suc);

  conn->used = true;
}

__device__ void zero_dup(Net *net, Cell *zero, Cell *dup, Connection *conn) {
  if (zero->symbol != ZERO || dup->symbol != DUP || zero->deleted ||
      dup->deleted) {
    return;
  }
  // zero main to dup main should clone 0 and connect it both
  // to whatever was connected to dup aux nodes
  Cell *new_zero = zero_cell_c(net);
  connect_c(zero->principal_port, dup->auxiliary_ports[0]->connected_to);
  connect_c(new_zero->principal_port, dup->auxiliary_ports[1]->connected_to);
  delete_cell_c(net, dup);

  conn->used = true;
  return;
}

__device__ void suc_dup(Net *net, Cell *suc, Cell *dup, Connection *conn) {
  if (suc->symbol != SUC || dup->symbol != DUP || suc->deleted ||
      dup->deleted) {
    return;
  }
  // if a suc main port connects to a dup main port, simply return
  // two sucs with main ports connected to the aux ports of dup
  // therefore we create a new suc and connect the old one
  Cell *new_suc = suc_cell_c(net);
  // connect dup principal port with suc aux conn
  connect_c(dup->principal_port, suc->auxiliary_ports[0]->connected_to);
  // conn suc principal_port to dup aux conn
  connect_c(suc->principal_port, dup->auxiliary_ports[0]->connected_to);
  connect_c(new_suc->principal_port, dup->auxiliary_ports[1]->connected_to);

  // connect suc and new suc aux ports with dup aux ports
  connect_c(suc->auxiliary_ports[0], dup->auxiliary_ports[0]);
  connect_c(new_suc->auxiliary_ports[0], dup->auxiliary_ports[1]);

  conn->used = true;
  return;
}

__device__ void suc_sum(Net *net, Cell *suc, Cell *sum, Connection *conn) {
  if (suc->symbol != SUC || sum->symbol != SUM || suc->deleted ||
      sum->deleted) {
    return;
  }
  // a suc port connected with a + port
  // makes the second aux port connected to a suc cell intermediately
  // and connects the + main port to whatever the suc aux port was connected
  Cell *new_suc = suc_cell_c(net);

  // first connect sum main port to s aux port conn
  connect_c(sum->principal_port, suc->auxiliary_ports[0]->connected_to);
  // conn the new suc main port to sum aux connected to
  connect_c(new_suc->principal_port, sum->auxiliary_ports[1]->connected_to);
  // conn the new suc auxiliar to the + second aux
  connect_c(new_suc->auxiliary_ports[0], sum->auxiliary_ports[1]);

  // now simply delete the old suc
  delete_cell_c(net, suc);

  // finally remove the connection from the array (maybe marking as used)
  conn->used = true;
}

__device__ void zero_sum(Net *net, Cell *zero, Cell *sum, Connection *conn) {
  if (zero->symbol != ZERO || sum->symbol != SUM || zero->deleted ||
      sum->deleted) {
    return;
  }
  // for this rule, we simply connect whatever is connected to 1st aux
  // port of the sum rule to whatever in the 2nd aux
  connect_c(sum->auxiliary_ports[0]->connected_to,
          sum->auxiliary_ports[1]->connected_to);
  // and delete both + and 0
  delete_cell_c(net, zero);
  delete_cell_c(net, sum);

  conn->used = true;
}

__device__ Rule mul_suc_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = mul_suc;
  rule.reducible = true;
  return rule;
}

__device__ Rule mul_zero_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = mul_zero;
  rule.reducible = true;
  return rule;
}

__device__ Rule zero_erasor_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = zero_erasor;
  rule.reducible = true;
  return rule;
}

__device__ Rule suc_erasor_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = suc_erasor;
  rule.reducible = true;
  return rule;
}

__device__ Rule zero_dup_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = zero_dup;
  rule.reducible = true;
  return rule;
}

__device__ Rule suc_dup_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = suc_dup;
  rule.reducible = true;
  return rule;
}

__device__ Rule suc_sum_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = suc_sum;
  rule.reducible = true;
  return rule;
}

__device__ Rule zero_sum_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = zero_sum;
  rule.reducible = true;
  return rule;
}

__device__ Rule do_nothing_rule(Connection *c) {
  Rule rule;
  rule.c = c;
  rule.reduce = do_nothing;
  rule.reducible = false;
  return rule;
}

__device__ void do_nothing(Net *net, Cell *a, Cell *b, Connection *conn) { return; }

__device__ Rule match_with_rule(Connection *c) {
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
// Rule find_reducible(Net *net) {
//   for (int i = 0; i < net->connection_count; i++) {
//     Connection *conn = net->connections[i];
//     if (conn->used) {
//       continue;
//     }
//     Rule r = match_with_rule(conn);
//     if (r.reducible) {
//       return r;
//     }
//   }
//   Rule null;
//   null.c = NULL;
//   null.reduce = do_nothing;
//   null.reducible = false;
//   return null;
// }

// ============== Specific Cell creation functions ================
Cell *suc_cell(Net *net) {
  Port *principal_port = create_port(true);
  Port *aux_ports[1];
  aux_ports[0] = create_port(false);
  Cell *suc = create_cell(SUC, principal_port, aux_ports, 1);
  add_cell_to_net(net, suc);
  return suc;
}

__device__ Cell *suc_cell_c(Net *net) {
  Port *principal_port = create_port_c(true);
  Port *aux_ports[1];
  aux_ports[0] = create_port_c(false);
  Cell *suc = create_cell_c(SUC, principal_port, aux_ports, 1);
  add_cell_to_net_c(net, suc);
  return suc;
}

Cell *zero_cell(Net *net) {
  Port *principal_port = create_port(true);
  Cell *zero = create_cell(ZERO, principal_port, NULL, 0);
  add_cell_to_net(net, zero);
  return zero;
}

__device__ Cell *zero_cell_c(Net *net) {
  Port *principal_port = create_port_c(true);
  Cell *zero = create_cell_c(ZERO, principal_port, NULL, 0);
  add_cell_to_net_c(net, zero);
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

__device__ Cell *dup_cell_c(Net *net) {
  Port *principal_port = create_port_c(true);
  Port *aux_ports[2];
  aux_ports[0] = create_port_c(false);
  aux_ports[1] = create_port_c(false);
  Cell *dup = create_cell_c(SUM, principal_port, aux_ports, 2);
  add_cell_to_net_c(net, dup);
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

__device__ Cell *sum_cell_c(Net *net) {
  // x
  Port *principal_port = create_port_c(true);
  Port *aux_ports[2];
  // y
  aux_ports[0] = create_port_c(false);
  // x + y
  aux_ports[1] = create_port_c(false);
  Cell *sum = create_cell_c(SUM, principal_port, aux_ports, 2);
  add_cell_to_net_c(net, sum);
  return sum;
}

Cell *erasor_cell(Net *net) {
  Port *principal_port = create_port(true);
  Cell *erasor = create_cell(ERA, principal_port, NULL, 0);
  add_cell_to_net(net, erasor);
  return erasor;
}

__device__ Cell *erasor_cell_c(Net *net) {
  Port *principal_port = create_port_c(true);
  Cell *erasor = create_cell_c(ERA, principal_port, NULL, 0);
  add_cell_to_net_c(net, erasor);
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

__device__ Cell *mul_cell_c(Net *net) {
  Port *principal_port = create_port_c(true);
  Port *aux_ports[2];
  aux_ports[0] = create_port_c(false);
  aux_ports[1] = create_port_c(false);
  Cell *dup = create_cell_c(MUL, principal_port, aux_ports, 2);
  add_cell_to_net_c(net, dup);
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

__device__ Port *create_port_c(bool is_principal) {
  Port *port = (Port *)malloc(sizeof(Port));
  port->is_principal = is_principal;
  port->connections = 0;
  port->connected_to = NULL;
  return port;
}


Cell *create_cell(Symbol symbol, Port *principal_port, Port **auxiliary_ports,
                  int aux_length) {
  Cell *c = (Cell *)malloc(sizeof(Cell));
  c->symbol = symbol;
  c->principal_port = principal_port;
  c->aux_ports_length = aux_length;
  c->deleted = false;
  c->auxiliary_ports = (Port **)malloc(aux_length * sizeof(Port *));
  for (int i = 0; i < aux_length; i++) {
    c->auxiliary_ports[i] = auxiliary_ports[i];
  }
  return c;
}

__device__ Cell *create_cell_c(Symbol symbol, Port *principal_port, Port **auxiliary_ports,
                  int aux_length) {
  Cell *c = (Cell *)malloc(sizeof(Cell));
  c->symbol = symbol;
  c->principal_port = principal_port;
  c->aux_ports_length = aux_length;
  c->deleted = false;
  c->auxiliary_ports = (Port **)malloc(aux_length * sizeof(Port *));
  for (int i = 0; i < aux_length; i++) {
    c->auxiliary_ports[i] = auxiliary_ports[i];
  }
  return c;
}

Net create_net() {
  Net n;
  n.connection_count = 0;
  n.cell_count = 0;
  n.cells = (Cell **)malloc(MAX_CONNECTIONS * sizeof(Cell *));
  n.connections = (Connection **)malloc(MAX_CONNECTIONS * sizeof(Connection *));
  return n;
}

__device__ void connect_c(Port *p, Port *q) {
  if (p == NULL || q == NULL) {
    return;
  }
  p->connected_to = q;
  p->connections = 1;
  q->connected_to = p;
  q->connections = 1;
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

__device__ void add_cell_to_net_c(Net *net, Cell *cell) {
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

__device__ void delete_cell_c(Net *net, Cell *cell) {
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

__device__ void do_nothinge(Net *net, Cell *a, Cell *b, Connection *conn) { return; }

__device__ bool check_mul_suce(Connection *c) {
  return (c->a->symbol == MUL && c->b->symbol == SUC) ||
         (c->a->symbol == SUC && c->b->symbol == MUL);
}


__device__ Rule match_with_rulee(Connection *c) {
  Rule rule;
  if (check_mul_suce(c)) {
    rule.c = c;
    rule.reduce = do_nothinge;
    rule.reducible = true;
    return rule;
  }
  rule.c = c;
  rule.reduce = do_nothinge;
  rule.reducible = false;
  return rule;
}

__global__ void find_reducible_kernel(Net *d_net, Rule *d_result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= d_net->connection_count) return;
  printf("idx: %i\n");
  Connection *conn = d_net->connections[idx];

  if (conn->used) return;

  Rule r = match_with_rulee(conn);
  if (r.reducible) {
    d_result[0] = r;
  }
}

Rule find_reducible(Net *net) {
  Net *d_net;
  Cell **d_cells;
  Connection **d_connections;
  Rule *d_result;
  cudaMalloc(&d_net, sizeof(Net));
  cudaMalloc(&d_cells, net->cell_count * sizeof(Cell *));
  cudaMalloc(&d_connections, net->connection_count * sizeof(Connection *));
  cudaMalloc(&d_result, sizeof(Rule));

  cudaMemcpy(d_net, net, sizeof(Net), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cells, net->cells, net->cell_count * sizeof(Cell *), cudaMemcpyHostToDevice);
  cudaMemcpy(d_connections, net->connections, net->connection_count * sizeof(Connection *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_net->cells), &d_cells, sizeof(Cell **), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_net->connections), &d_connections, sizeof(Connection **), cudaMemcpyHostToDevice);

  printf("Conn count: %i\n", d_net->connection_count)

  int threadsPerBlock = 256;
  int blocksPerGrid = (net->connection_count + threadsPerBlock - 1) / threadsPerBlock;
  find_reducible_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_net, d_result);

  cudaDeviceSynchronize();

  // printf("IS cuda rule result null? %i\n", d_result == NULL);
  // printf("Cuda rule: is reducible?%i\n", d_result->reducible);

  Rule h_result;
  cudaMemcpy(&h_result, d_result, sizeof(Rule), cudaMemcpyDeviceToHost);

  cudaFree(d_cells);
  cudaFree(d_connections);
  cudaFree(d_net);
  cudaFree(d_result);

  return h_result;
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
  // r.reduce(&net, r.c->a, r.c->b, &conn);
  // print_net(&net);
  // update_connections(&net);

  // Rule r2 = find_reducible(&net);
  // print_rule(&r2);
  // r2.reduce(&net, r2.c->a, r2.c->b, r2.c);
  // print_net(&net);

  // update_connections(&net);

  // Rule r3 = find_reducible(&net);
  // printf("Is r3 a non reducible null rull (halted?) %i\n",
  //        r3.reducible == false);
}
