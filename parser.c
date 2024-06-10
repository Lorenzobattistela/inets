#include "parser.h"

TokenInfo current_token;

TokenInfo get_next_token(const char **input) {
  while (**input == ' ')
    (*input)++; // skip whitespace
  if (**input == '\0')
    return (TokenInfo){END, 0};

  if (**input == '(') {
    (*input)++;
    return (TokenInfo){L_PAREN, 0};
  }
  if (**input == ')') {
    (*input)++;
    return (TokenInfo){R_PAREN, 0};
  }
  if (**input == '*') {
    (*input)++;
    return (TokenInfo){MULTIPLICATION, 0};
  }
  if (**input == '+') {
    (*input)++;
    return (TokenInfo){PLUS, 0};
  }
  if (isdigit(**input)) {
    int value = 0;
    while (isdigit(**input)) {
      value = value * 10 + (**input - '0');
      (*input)++;
    }
    return (TokenInfo){DIGIT, value};
  }
  return (TokenInfo){END, 0};
}

void advance(const char **input) { current_token = get_next_token(input); }

void match(Token expected, const char **input) {
  if (current_token.token == expected) {
    advance(input);
  } else {
    printf("Syntax error: expected %d, found %d\n", expected,
           current_token.token);
    exit(1);
  }
}

ASTNode *create_ast_node(Token token, int value, ASTNode *left,
                         ASTNode *right) {
  ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
  node->token = token;
  node->value = value;
  node->left = left;
  node->right = right;
  return node;
}

ASTNode *parse_expression(const char **input);
ASTNode *parse_term(const char **input);

ASTNode *parse_term(const char **input) {
  ASTNode *node;
  if (current_token.token == DIGIT) {
    node = create_ast_node(DIGIT, current_token.value, NULL, NULL);
    advance(input);
  } else if (current_token.token == L_PAREN) {
    advance(input);
    node = parse_expression(input);
    match(R_PAREN, input);
  } else {
    printf("Syntax error: unexpected token %d\n", current_token.token);
    exit(1);
  }
  return node;
}

ASTNode *parse_expression(const char **input) {
  ASTNode *node = parse_term(input);
  while (current_token.token == PLUS || current_token.token == MULTIPLICATION) {
    Token op = current_token.token;
    advance(input);
    ASTNode *right = parse_term(input);
    node = create_ast_node(op, 0, node, right);
  }
  return node;
}

ASTNode *parse(const char *input) {
  advance(&input);
  return parse_expression(&input);
}

void print_ast(ASTNode *node) {
  if (node == NULL)
    return;
  if (node->token == DIGIT) {
    printf("%d", node->value);
  } else if (node->token == MULTIPLICATION) {
    printf("(");
    print_ast(node->left);
    printf(" * ");
    print_ast(node->right);
    printf(")");
  } else if (node->token == PLUS) {
    printf("(");
    print_ast(node->left);
    printf(" + ");
    print_ast(node->right);
    printf(")");
  }
}

void print_token(Token t) {
  switch (t) {
  case DIGIT:
    printf("DIGIT\n");
    break;
  case PLUS:
    printf("PLUS\n");
    break;
  default:
    break;
  }
}

void free_ast(ASTNode *node) {
  if (node == NULL)
    return;
  free_ast(node->left);
  free_ast(node->right);
  free(node);
}

// int main() {
//   const char *input = "((3 + 4) + (2 + 3))";
//   ASTNode *ast = parse(input);
//   printf("AST: ");
//   print_ast(ast);
//   printf("\n");
//   free_ast(ast);
//   return 0;
// }
