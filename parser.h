#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

ASTNode *parse(const char *input);
void free_ast(ASTNode *node);
void print_token(Token t);
