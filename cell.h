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
typedef struct Connection Connection;
typedef void (*FunctionPtr)(Net *, Cell *, Cell *, Connection *);

typedef enum {
  ERA,  // erasor
  DUP,  // duplicator
  SUC,  // successor
  ZERO, // zero
  SUM,  // sum
  MUL,  // multiplication
  ANY,
  NULL_S,
} Symbol;

struct Port {
  bool is_principal;
  Port *connected_to;
  int connections;
};

struct Cell {
  Symbol symbol;
  Port **auxiliary_ports;
  int aux_ports_length;
  Port *principal_port;
  bool deleted;
};

struct Rule {
  Connection *c;
  FunctionPtr reduce;
  bool reducible;
};

struct Connection {
  Cell *a;
  Cell *b;
  bool used;
};

struct Net {
  Cell **cells;
  int cell_count;
  Connection **connections;
  int connection_count;
};

Rule match_with_rule(Connection *c);
Rule *find_reducible(Net *net);

// specific cell creating functions
Cell *suc_cell(Net *net);
Cell *zero_cell(Net *net);
Cell *dup_cell(Net *net);
Cell *sum_cell(Net *net);
Cell *erasor_cell(Net *net);
Cell *mul_cell(Net *net);

// Cell interactions fns
Cell *clone_cell(Cell *cell);
Port *create_port(bool is_principal);
Cell *create_cell(Symbol symbol, Port *principal_port, Port *auxiliary_ports[],
                  int aux_length);
void connect(Port *p, Port *q);
void erase(Net *net, Cell *erasor, Cell *to_erase);
void add_cell_to_net(Net *net, Cell *cell);
void delete_cell(Net *net, Cell *cell);

// pretty printing functions
void print_net(Net *net);
void print_port(Port *p);
void pprint_symbol(Symbol s);
void print_cell(Cell *cell);

Rule dup_rule();
Rule erase_rule();

void mul_suc(Net *net, Cell *mul, Cell *suc, Connection *conn);
void mul_zero(Net *net, Cell *mul, Cell *zero, Connection *conn);
void zero_erasor(Net *net, Cell *zero, Cell *erasor, Connection *conn);
void suc_erasor(Net *net, Cell *suc, Cell *erasor, Connection *conn);
void zero_dup(Net *net, Cell *zero, Cell *dup, Connection *conn);
void suc_dup(Net *net, Cell *suc, Cell *dup, Connection *conn);
void suc_sum(Net *net, Cell *suc, Cell *sum, Connection *conn);
void zero_sum(Net *net, Cell *zero, Cell *sum, Connection *conn);
void do_nothing(Net *net, Cell *a, Cell *b, Connection *conn);
