import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def matrix_multiplication_with_blocking(matrix_size, block_size):
    m1 = np.random.rand(matrix_size, matrix_size)
    m2 = np.random.rand(matrix_size, matrix_size)
    # intialize zero matrix same size of m1 & m2
    res = np.zeros((matrix_size, matrix_size))

    for i in range(0, matrix_size, block_size):
        for j in range(0, matrix_size, block_size):
            for k in range(0, matrix_size, block_size):
                m1_block = m1[i:i+block_size, k:k+block_size]
                m2_block = m2[k:k+block_size, j:j+block_size]
                res[i:i+block_size, j:j+block_size] += np.dot(m1_block, m2_block)
    return res

def matrix_multiplication_without_blocking(matrix_size):
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)
    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C

matrix_sizes = [64,128,256,512]
block_sizes = [1,4,8,16,32]

results = []
for matrix_size in matrix_sizes:
    for block_size in block_sizes:
        start_time = time.time()
        matrix_multiplication_with_blocking(matrix_size, block_size)
        end_time = time.time()
        time_taken = end_time - start_time
        results.append((matrix_size, block_size, time_taken))

df = pd.DataFrame(results, columns=['Matrix Size', 'Block Size', 'Time Taken'])
# Create a plot
fig, ax = plt.subplots(figsize=(10,10))

for matrix_size in matrix_sizes:
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Figure 1')
    ax.plot(df[df['Matrix Size']==matrix_size]['Block Size'], df[df['Matrix Size']==matrix_size]['Time Taken'], label='Matrix Size = ' + str(matrix_size))

ax.legend()
plt.show()

# crate a table
table = pd.pivot_table(df, values='Time Taken', index=['Matrix Size'], columns=['Block Size'])
table_formatted = table.applymap(lambda x: '{:.4f}'.format(x))
fig2 = plt.figure(figsize=(12,8))
plt.axis('off')
plt.title('Figure 2\nMatrix Multiplication with Blocking Results', fontsize=12)
plt.table(cellText=table_formatted.values, colLabels=table_formatted.columns, rowLabels=table_formatted.index, loc='center', fontsize=20)
plt.show()

#main for matrix multiplication without blocking


results = []
for matrix_size in matrix_sizes:
    start_time = time.time()
    matrix_multiplication_without_blocking(matrix_size)
    end_time = time.time()
    time_taken = end_time - start_time
    results.append((matrix_size, time_taken))

df = pd.DataFrame(results, columns=['Matrix Size', 'Time Taken'])

# Create a plot
plt.plot(df['Matrix Size'], df['Time Taken'])
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (s)')
plt.title('Figure 1')
plt.show()

# Create a table
fig, ax = plt.subplots(figsize=(8, 6))
table_values = df.applymap(lambda x: '{:.4f}'.format(x)).values
table = ax.table(cellText=table_values, colLabels=df.columns, loc='center')
ax.set_title('Figure 2')
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Execution Time (s)')
ax.axis('off')
plt.show()
