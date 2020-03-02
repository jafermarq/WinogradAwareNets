import numpy as np

# [1] "On Improving the Numerical Stability of Winograd Convolutions", Vincent et al. (ICLRw, 2018)
# [2] "Fast Algorithms for Convolutional Neural Networks", Lavin & Gray (CVPR, 2016)
# [3] "Improving accuracy of winiograd convolutions for dnns", Barabasz et al. (2018)

def construct_vandermonde(rows: int, cols: int, poly_points: list):
    """ Constructs Vandermonde matrix as described in [1] 
        Args:
            rows (int): rows of matrix
            cols (int): columns of matrix
            polyPoints (list): polynomial points used for matrix generation (x,y) format
        Return:
            V (np.ndarray): rows x cols Vandermonde matrix
    """

    assert rows == len(poly_points)

    for r in range(rows):
        row = []
        for c in range(cols):
            f = poly_points[r][0]
            g = poly_points[r][1]
            row.append(f**c * g**(cols - c -1))
        
        if r == 0:
            V = np.array(row)
        else:
            V = np.vstack([V, row])

    return V.astype(np.float32)


def get_winograd_transforms(m: int, n: int, diagonals: list, poly_points: list) -> np.ndarray:
    """ Generates Winograd (Cook-Toom) transformation matrices given:
        Args:
            m (int): the number of outputs
            n (int): the kernle size
            diagonals (list): a list of values for diagonal matrices Sy, Sx, Sw
            polyPoints (list): a list of polynomial points for Vandermonde matrix generation
        Returns:
            Y_t (np.ndarray): final transform matrix to original space
            X_t (np.ndarray): data transformation matrix to Winograd space
            W (np.ndarray): kernel trasnformation matrix to Winograd space
    """

    Sy = np.diag(diagonals[0]).astype(np.float32)
    Sx = np.diag(diagonals[1]).astype(np.float32)
    Sw = np.diag(diagonals[2]).astype(np.float32)

    V_full_m = construct_vandermonde(m+n-1,m, poly_points)
    A_t = np.dot(np.transpose(V_full_m), Sy) # Y_t in [1]

    V_sqr = construct_vandermonde(m+n-1, m+n-1, poly_points)
    V_sqrt_inv = np.linalg.inv(V_sqr)
    B_t = np.dot(Sx, np.transpose(V_sqrt_inv)) # X_t in [1]

    V_full_n = construct_vandermonde(m+n-1, n, poly_points)
    G = np.dot(Sw, V_full_n) # W in [1]

    return A_t, B_t, G


def get_transforms(m: int, n: int):
    ''' Given number of outpus and kernel size, construct the Winograd transformation matrices (A,B,G) as described in [1] '''

    if m == 2 and n == 3: #generate transforms A,B,G for F[2x2, 3x3] as in [2] 
        diagonals = [[1,1,1,-1],[1,2,2,-1],[1,0.5,0.5,1]]
        polyPoints = [[0,1],[1,1],[-1,1],[1,0]]

    elif m == 4 and n == 3: #generate transforms A,B,G for F[4x4, 3x3] as in [2] 
        diagonals = [[1, 1, 1, 1, 1, 1],[4, -6, -6, 24, 24, 1],[1/4,-1/6,-1/6,1/24,1/24,1]]
        polyPoints = [[0,1],[1,1],[-1,1],[2,1],[-2,1],[1,0]]

    elif m == 6 and n == 3: # empirically found F(6x6, 3x3) parameters from F[4x4, 3x3] parameters and recipe in [3]
        diagonals = [[1, 1, 1, 1, 1, 1, 1, 1],[4, -6, -6, 24, 24, 1, -12, -12],[1/4,-1/6,-1/6,1/24,1/24, -1/12,  -1/12, 1]]
        polyPoints = [[0,1],[1,1],[-1,1],[2,1],[-2,1], [3/2,1], [-3/2,1], [1,0]]
    else:
        raise ValueError('Need to define parameters for F(' + str(m) + ' x ' + str(m) + ','  + str(n) + ' x ' + str(n) + ')')

    A_t, B_t, G = get_winograd_transforms(m, n, diagonals, polyPoints)

    return A_t, B_t, G


class Tiler:
    ''' Winograd convolutions require to split the input into overlaped tiles. We use this class to perform the tiling (at the begining of the winogradAware layers) of the input and untiling of the output (right before leaving the `forward()` pass of the Winograd-aware layers)'''

    def __init__(self, F: int, filterDim: int):
        self.F = F
        self.chunk_dim = self.F + filterDim - 1
        self.stride = F
        self.tiledShape = 0
        self.numChunks = 0

    def tile(self, input):

        tensor = input.unfold(2, self.chunk_dim, self.stride).unfold(3, self.chunk_dim, self.stride)

        self.tiledShape = tensor.shape
        self.numChunks = tensor.shape[2]
        return tensor.contiguous().view(tensor.size(0), tensor.size(1), -1, tensor.size(4), tensor.size(5))


    def untile(self, output):
        output = output.reshape(output.shape[0], output.shape[1], self.tiledShape[2], self.tiledShape[3], self.F, self.F)
        return output.transpose(4,3).contiguous().squeeze().view(output.shape[0], output.shape[1], self.F*self.numChunks, self.F*self.numChunks)