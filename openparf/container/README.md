\ingroup GrpDummyPages
\page OpenPARF_Container Container

Basic container classes such as flat nested vector to represent array of arrays.

# Flat Nested Array

Flat nested array is widely used to convert nested array into flat data structures for better support of hardware acceleration kernels.
It is essentially a simplified version of the compressed sparse matrix (CSR) format.
The data structure introduces two arrays to represent the nested information.
One array stores all the elements and the other stores the begin index of each inner array.

![Flat Nested Array Representation](@ref flat_nested_array.svg "flat_nested_array.svg")
@image latex flat_nested_array.pdf "Flat Nested Array Representation" width=\textwidth
