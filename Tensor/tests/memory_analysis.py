""""
Here we analyze memory layout and performance
"""
from tensor import BYTES_PER_FLOAT32, MB_TO_BYTES, Tensor
import numpy as np
def analyze_memory_layout():
    """Demonstrate cache effect with rows vs column access patterns."""
    print("Analyzing Memory Access patterns")
    print("="*60)

    #creating a moderately-sized matrix
    size = 2000
    matrix = Tensor(np.random.rand(size,size))

    print(f"\nTesting with {size}x{size} matrix ({matrix.size * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB)")
    print("-"*60)

    import time

    #testing row-wise access
    #memory layout [row0][row1][row2]... stored contigously
    print("\n Test 1: Row-wise Access")
    start =time.time()
    row_sums = []

    for i in range(size):
        row_sum = matrix.data[i,:].sum() #accessing entire row sequentially
        row_sums.append(row_sum)
    row_time = time.time() - start
    print(f"   Time: {row_time*1000:.1f}ms")
    print(f"   Access pattern: Sequential (follows memory layout)")


    #2. Testing Column-wise access
    #must jump between rows, poor spatial locality
    print("\n Test 2: Column-wise Access (Cache-unfriendly)")
    start = time.time()
    col_sums = []

    for j in range(size):
        col_sum = matrix.data[:,j].sum()
        col_sums.append(col_sum)
    col_time = time.time() - start
    print(f"  Time:{col_time*1000:.1f}ms")
    print(f" Access pattern: Strided (jumps {size*BYTES_PER_FLOAT32} bytes per elements)")

    #calculating slowdown
    slowdown = col_time / row_time
    print("\n" + "="*60)
    print(f"Performance Impact: ")
    print(f" Slowdown factor: {slowdown:.2f}x ({col_time/row_time:.1f}x slower)")
    print(f" Cache misses cause {(slowdown-1)*100:.0f}% performance loss")


if __name__ == "__main__":
    analyze_memory_layout()
