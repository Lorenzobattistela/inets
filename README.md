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

At first I decided to make `find_reducible` parallel. The reason is because it searches for active pairs through the whole net, so each thread can look up two cells and tell if they're reducible or not.

After doing this, I still used the sequential way to reduce the nets. This makes the program lazy as hell, because we keep calling the kernel and all the memcpy operations happen for every reduction (if we have 4000 reductions we will do all the allocation and freeing 4000 times, this looks really bad). Thats why the perfomance drops and its a lot slower than the C sequential implementation.
 
The next step is to make reductions parallel as well. We only have to be careful about ports linking. Since i have to make array copying everywhere (because I used C structs, and deep copies are not a good thing to do in CUDA), ill have to pass the ports of the cells as well so i can make the linking and then recopying into the old struct (or simply ignore all the structs and just use arrays everywhere, which would make it faster). Linking operations also have to be atomic, so we dont try to change a port if other thread is reducing it. So when linking two ports, both are locked by the atomic op.

The result in the CPU function would be the reduced net. We can implement this step-wise and then add the while loop in the gpu as well, performing all the reductions there. This way the mem copying would happen only once in the beginning of the program and another time to give back the result.

The reduction process would need one thread per redex.

To process the whole net in the gpu, we'd have to generalize the kernel caller. Then we would need one kernel for finding reducible and another kernel to reduce. The former is solved already.

### Reduce kernel:
- We need an array of arrays of ints. (int **). This should hold the ports for a given cell. The array of ports of the cell with id 1 is obviously at ports[1]. This will be used on the linking step.
- Rewrite the reduction functions as device, as well as the link function with atomic operations. Also the delete_cell, since we would have our GPU net as well.
- Rewrite cell creation specifically to gpu. Atomic add to increase the cell_counter (shared memory).

## Old approach

When I first tried to implement inets, I tried something a bit more expressive in terms of legibility (using more structs), but it turns out to be a lot more complicated and really slow. The implementation is at `old_approach/cell.c` and `unary_arithmetics.c`. Not really easy to make it parallel the way it was wrote. But it works.


## Working version improvements
- pre-alloc cells instead of alloc them everytime i create
- instead of having my own "net structure" and convert it to arrays to process in the gpu, do it all with the same structure
- valgrind to catch memory leaks
- not alloc everything on MAX_CELLS for all arrays, doing cell_count
- link should be atomic. Same program giving different results arbitrairly