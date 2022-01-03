import numpy as np

# dimension of the (square) matrix
n = 20
# whether to show progress on the command line
show_progress = True
ITER_LIMIT = 10000

def progress_bar(iteration, total, length = 60, clear = True):
    if not show_progress:
        return
    if clear:
        # clear the current line to get rid of the progress bar
        print(end = "\033[2K")
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '=' * filledLength + ' ' * (length - filledLength)
    print(f'\r[{bar}] {percent}%', end = "\r")
    if iteration == total:
        print()

def generate_input(gamma):
    # initialize a matrix with gammas on the diagonal,
    # -1 below and above the diagonal, and zero everywhere else
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = gamma
        if i > 0:
            A[i - 1, i] = -1
        if i < n - 1:
            A[i + 1, i] = -1
    # initialize a vector of size n with gamma - 2 everywhere
    # except for gamma - 1 at the first and last index
    b = np.zeros(n)
    b[0] = gamma - 2
    b[n - 1] = gamma - 2
    return A, b

# Get the diagonal component of a square matrix
def diagonal_component(A: np.ndarray) -> np.ndarray:
    diag = np.zeros(A.shape)
    for i in range(np.shape(A)[0]):
        diag[i, i] = A[i, i]
    return diag

def process_jacobi_constants(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = diagonal_component(A)
    Dinv = np.zeros(D.shape)
    for i in range(D.shape[0]):
        Dinv[i, i] = 1 / D[i, i]

    T = -Dinv.dot(A - D)
    C =  Dinv.dot(b)
    return (T, C)

def jacobi(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    progress_bar(0, ITER_LIMIT, clear = False)
    # the result of the previous iteration
    last = None
    # the result of the current iteration
    next = np.zeros(n)
    (T, C) = process_jacobi_constants(A, b)

    for iteration in range(ITER_LIMIT):
        last = next
        next = T.dot(last) + C

        # check the convergence criterion
        if np.linalg.norm(A.dot(next) - b) / np.linalg.norm(b) < 10 ** -6:
            print("\ndone @ iteration", iteration)
            break

        if iteration % 10 == 0 or iteration > ITER_LIMIT - 10:
            progress_bar(iteration, ITER_LIMIT - 1)

    print()
    return next

(A, b) = generate_input(5)
print(jacobi(A, b))
