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
typedef void (*FunctionPtr)(Net *, Cell *, Cell *);

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
  Port *auxiliary_ports[MAX_AUX_PORTS];
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
};

struct Net {
  Cell *cells[MAX_CONNECTIONS];
  int cell_count;
  Connection *connections[MAX_CONNECTIONS];
  int connection_count;
};

Rule match_with_rule(Connection *c);
Rule find_reducible(Net *net);

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

void mul_suc(Net *net, Cell *mul, Cell *suc);
void mul_zero(Net *net, Cell *mul, Cell *zero);
void zero_erasor(Net *net, Cell *zero, Cell *erasor);
void suc_erasor(Net *net, Cell *suc, Cell *erasor);
void zero_dup(Net *net, Cell *zero, Cell *dup);
void suc_dup(Net *net, Cell *suc, Cell *dup);
void suc_sum(Net *net, Cell *suc, Cell *sum);
void zero_sum(Net *net, Cell *zero, Cell *sum);
void do_nothing(Net *net, Cell *a, Cell *b);
