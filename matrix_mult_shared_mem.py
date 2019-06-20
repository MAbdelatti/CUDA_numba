import numpy as np
from numba import cuda, types

# Leave the values in this cell alone
M = 128
N = 32

# Input vectors of MxN and NxM dimensions
a = np.arange(M*N).reshape(M,N).astype(np.int32) # 128x32
b = np.arange(M*N).reshape(N,M).astype(np.int32) # 32X128
c = np.zeros((M, M)).astype(np.int32)            # 128x128

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

# NxN threads per block, in 2 dimensions
block_size = (N,N) # 32x32
# MxM/NxN blocks per grid, in 2 dimensions
grid_size = (int(M/N),int(M/N)) # 4x4

@cuda.jit(debug=True)
def mm_shared(a, b, c):
    row, column = cuda.grid(2) # 0 - 127 , 0 - 127
    
    sum = 0
    tx = cuda.threadIdx.x # 0 - 32
    ty = cuda.threadIdx.y # 0 - 32
    
    # `a_cache` and `b_cache` are already correctly defined
    a_cache = cuda.shared.array(block_size, types.int32) # 32x32 although a is 128x32
    b_cache = cuda.shared.array(block_size, types.int32) # 32x32 although b is 32x128
    
    # TODO: use each thread to populate one element each a_cache and b_cache       
    for i in range(grid_size[0]): # 32
        a_cache[tx, ty] = a[row, ty + i * N]
        b_cache[tx, ty] = b[tx + i * N, column]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # TODO: calculate the `sum` value correctly using values from the cache 
        for j in range(N):
            sum += a_cache[tx, j] * b_cache[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    c[row, column ] = sum

# There's no need to update this kernel launch
mm_shared[grid_size, block_size](d_a, d_b, d_c)

# Do not modify the contents in this cell
from numpy import testing
solution = a@b
print(solution)
output = d_c.copy_to_host()
print(output)
# This assertion will fail until you correctly update the kernel above.
testing.assert_array_equal(output, solution)
