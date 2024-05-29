#include <stdio.h>
#include <stdlib.h>
#include "parser.h"

#define MAX_CELLS 3500

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

typedef void (*ReductionFunc)(Cell **, Cell *, Cell *);

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
    int a_id, b_id;

    while(find_reducible(net, &reduce_function, &a_id, &b_id)) {
        if(net[a_id]->type == SUM && net[b_id]->type == SUC) {
            reduce_function(net, net[b_id], net[a_id]);
        } else if (net[a_id]->type == SUM && net[b_id]->type == ZERO) {
            reduce_function(net, net[b_id], net[a_id]);
        } else {
            reduce_function(net, net[a_id], net[b_id]);
        }
    }

    int val = church_decode(net);
    printf("Decoded value: %d\n", val);
    
    for (int i = 0; i < MAX_CELLS; ++i) {
        if (net[i] != NULL) {
            delete_cell(net, i);
        }
    }
    free_ast(ast);
    return 0;   
}
