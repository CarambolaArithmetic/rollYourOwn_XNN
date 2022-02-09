Since the convolution demo has been fixed, here are the things that are left to do:
1. Restyle node, math, and network.py to be more presentable:
    * renaming symbols according to PEP8 standards (style branch)
    * creating cleaner, better organized comments
    * removing or refactoring extraneous functions
3. Write Unit Tests for Nodes to be used as regression testing, particularly for the purpose of doing performance testing
2. Fix performance:
    * write perf tests
    * improve calling structure of forward and backward passes to reduce depth of call stacks
        * idea: put operations into a graph of functions and use an accumulator to bring data between them
        * (I do not think that this will change performance much)
    * add shape parameters to Nodes to allow exploring how various ways of organizing the input data affects performance 
        * do this through replacing most numpy operations with einsum
    * try a direct comparison of einsum and matmul, and other stuff
     * try other forms of the convolution operator that do not require using einsum at all.