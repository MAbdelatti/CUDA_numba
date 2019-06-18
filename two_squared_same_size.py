''' Since the matrices a and b are squared and have the same size equal to the cached memory,
the solution is straight forward and does not require tiles'''

import numpy as np
from numba import cuda, types

# Leave the values in this cell alone
M = 32
N = 32

# Input vectors of MxN and NxM dimensions
a = np.arange(M*N).reshape(M,N).astype(np.int32) # 32x32
b = np.arange(M*N).reshape(N,M).astype(np.int32) # 32X32
c = np.zeros((M, M)).astype(np.int32)            # 32x128

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

# NxN threads per block, in 2 dimensions
block_size = (N,N) # 32x32
# MxM/NxN blocks per grid, in 2 dimensions
grid_size = (int(M/N),int(M/N)) # 4x4

# ------------------------------------------------
# Works very well when a and b are squared matrices with the same dimensions:

@cuda.jit
def mm_shared(a, b, c):
    column, row = cuda.grid(2)
    sum = 0

    # `a_cache` and `b_cache` are already correctly defined
    a_cache = cuda.shared.array(block_size, types.int32)
    b_cache = cuda.shared.array(block_size, types.int32)

    # TODO: use each thread to populate one element each a_cache and b_cache
    a_cache[column, row] = a[column, row]
    b_cache[column, row] = b[column, row]

    
    for i in range(a.shape[1]):
        # TODO: calculate the `sum` value correctly using values from the cache 
        sum += a_cache[i][column] * b_cache[row][i]
        
    c[row][column] = sum

# -------------------------------------------------
# There's no need to update this kernel launch
mm_shared[grid_size, block_size](d_a, d_b, d_c)

# -------------------------------------------------
# Do not modify the contents in this cell
from numpy import testing
solution = a@b
print(solution)
output = d_c.copy_to_host()
print(output)
# This assertion will fail until you correctly update the kernel above.
testing.assert_array_equal(output, solution)
