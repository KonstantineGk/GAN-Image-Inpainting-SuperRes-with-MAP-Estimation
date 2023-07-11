import numpy as np

# Transformation matrix for Super Resolution
def create_T():
    T = np.zeros((49, 784))
    for i in range(7):
        for j in range(7):
            row_start = i * 4
            row_end = (i * 4) + 4
            col_start = j * 4
            col_end = (j * 4) + 4
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    T[i * 7 + j, r * 28 + c] = 1 / 16
    return T