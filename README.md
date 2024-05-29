# Inets evaluator

This is an interaction nets evaluator. For now it only modelates simple sum (but is fairly simple to be extended with more connectors).

## Implementation

The current implementation is in the files `inets.c` and `parser.c` . The parser is a simple helper to write expressions as `(1 + 1)` and turn them into an interaction net.

For this simple non production implementation, I chose to represent each Cell type using ints (e.g ZERO = 0, SUC = 1) and rules as the sum of the ints. Note that this is not a good approach, since the sum 1 + 1 is equal to 0 + 2. But i have 3 rules so i would not overcomplicate now for this, first lets get this working. But a simple solution would be to have a mask for each type.

In cells, we hold its type and an array of ports (which are Port structs). For this example, i also used fixed-sized arrays, otherwise i'd had to deal with dyn arrays implementation and the focus is CUDA after the C impl.

Each Port holds the connected_cell id and the connected_port id. The cell id and port id are simply their index in the array.

Link operations simply change the Port connections values.

Then we `to_net` to transform an AST in a Net, an array of cells. 

We iteractively - and this can be done in parallel - find active pairs (reducible ones) and apply the rule returned untill there is no more active pairs with valid rules. Since I just have addition, i added a church decode function to show the result after the process.

To run the C implementation, simply run `make` in the root dir and then `./inets` . You should see 8 as the decoded value. You can change the parsed string in `inets.c` and recompile it with make.

I also wrote a simple POC for my implementation idea in python, so `nets.py` should work as well.

## Using CUDA to make it parallel

WIP


