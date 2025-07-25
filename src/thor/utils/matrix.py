import numpy as np

def ld_entry(matrix:np.ndarray, l:int, i:int):
    """
    Get the i th entry of the l th lower diagonal of the matrix
    """
    b, c = matrix.shape 
    return matrix[(l + i) %b, i % c]

def ud_entry(matrix:np.ndarray, u:int, i:int, rot:int=0):
    """
    Get the i th entry of the u th upper diagonal of the matrix
    """
    i = i + rot
    a, b = matrix.shape 
    return matrix[i %a, (u+i) % b]

def ud(matrix:np.ndarray, u:int):
    """
    Get the u th upper diagonal of the matrix
    """
    a, b = matrix.shape
    dim = max(a,b)
    return np.array([ud_entry(matrix, u, i) for i in range(dim)])

def ld(matrix:np.ndarray, l:int):
    """
    Get the l th lower diagonal of the matrix
    """
    a, b = matrix.shape
    dim = max(a,b)
    return np.array([ld_entry(matrix, l, i) for i in range(dim)])

def to_blocks(matrix:np.ndarray, block_shape:tuple[int], diag:bool=True) -> tuple[np.ndarray, tuple[int]]:
    """
    Convert the matrix to a list of block matrices.
    Return the blocks in diagonal form if diag is True
    """
    a, b = matrix.shape
    if a % block_shape[0] != 0 or b % block_shape[1] != 0:
        raise ValueError("Matrix shape should be divisible by block shape")
    v = a // block_shape[0]
    h = b // block_shape[1]
    blocks = np.empty((v, h), dtype=object)
    row_blocks = np.vsplit(matrix, v)
    for i, row_block in enumerate(row_blocks):
        blocks[i] = np.hsplit(row_block, h)
    if not diag:
        return blocks, (v, h)
    else:
        l = min(v, h)
        d = max(v, h)
        diag_blocks = np.empty((l, d), dtype=object)
        for t in range(d):
            for i in range(l):
                diag_blocks[i, t] = blocks[(i+t)%v, t%h]
        return diag_blocks, (l, d)

def diag_blocks(blocks:list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """
    Convert the list of block matrices to a matrix
    """
    v = len(blocks)
    h = len(blocks[0])
    block_shape = blocks[0][0].shape
    a = v * block_shape[0]
    b = h * block_shape[1]
    matrix = np.zeros((a, b))
    for i in range(v):
        for j in range(h):
            matrix[i*block_shape[0]:(i+1)*block_shape[0], j*block_shape[1]:(j+1)*block_shape[1]] = blocks[i][j]
    return matrix


