#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CELLS 3500

#define SUM 0
#define SUC 1
#define ZERO 2

#define SUC_SUM (SUM + SUC)
#define ZERO_SUM (ZERO + SUM)

typedef enum { L_PAREN, R_PAREN, DIGIT, PLUS, END } Token;

typedef struct {
  Token token;
  int value;
} TokenInfo;

typedef struct ASTNode {
  Token token;
  int value;
  struct ASTNode *left;
  struct ASTNode *right;
} ASTNode;

TokenInfo get_next_token(const char **input);
void advance(const char **input);
ASTNode *create_ast_node(Token token, int value, ASTNode *left, ASTNode *right);
ASTNode *parse_expression(const char **input);
ASTNode *parse_term(const char **input);
ASTNode *parse(const char *input);
void print_ast(ASTNode *node);
void print_token(Token t);
void free_ast(ASTNode *node);

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

typedef void (*ReductionFunc)(Cell **, Cell *, Cell *);

Cell* create_cell(int cell_type, int num_aux_ports);
void delete_cell(Cell **cells, int cell_id);
void add_to_net(Cell **net, Cell *cell);
Cell* zero_cell(Cell **net);
Cell* suc_cell(Cell **net);
Cell* sum_cell(Cell **net);
void link(Cell **cells, int a, int a_idx, int b, int b_idx);
void suc_sum(Cell **cells, Cell *suc, Cell *s);
void zero_sum(Cell **cells, Cell *zero, Cell *s);
int check_rule(Cell *cell_a, Cell *cell_b, ReductionFunc *reduction_func);
int find_reducible(Cell **cells, ReductionFunc *reduction_func, int *a_id, int *b_id);
int church_encode(Cell **net, int num);
Cell* find_zero_cell(Cell **net);
int church_decode(Cell **net);
int to_net(Cell **net, ASTNode *node);