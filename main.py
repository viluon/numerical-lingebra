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
    print(f'\r  [{bar}] {percent}%', end = "\r")
    if iteration == total:
        print()

def generate_input(gamma: float):
    # initialize a matrix with gammas on the diagonal,
    # -1 below and above the diagonal, and zero everywhere else
    A = np.zeros((n, n))
    # initialize a vector of size n with gamma - 2 everywhere
    # except for gamma - 1 at the first and last index
    b = np.zeros(n)
    for i in range(n):
        A[i, i] = gamma
        b[i] = gamma - 2
        if i > 0:
            A[i - 1, i] = -1
        if i < n - 1:
            A[i + 1, i] = -1
    b[0] = gamma - 1
    b[n - 1] = gamma - 1
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

def jacobi_wrong(A: np.ndarray, b: np.ndarray) -> np.ndarray:
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

def split_LDU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = np.zeros(A.shape)
    D = np.zeros(A.shape)
    U = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            target = D if i == j else (L if i < j else U)
            target[i, j] = A[i, j]

    assert (L + D + U == A).all()
    return (L, D, U)

def jacobi(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    (L, D, U) = split_LDU(A)
    return D

def gauss_seidel(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    (L, D, U) = split_LDU(A)
    return D + L

def iteratively(method, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    progress_bar(0, ITER_LIMIT, clear = False)
    # the result of the previous iteration
    last = None
    # the result of the current iteration (starting with the zero vector)
    next = np.zeros(n)
    # use the Euclidean distance
    norm = lambda v: np.linalg.norm(v, ord = 2)

    b_norm = norm(b)
    Q = method(A, b)
    Q_minus_A = Q - A

    # compute Q^(-1)
    Qinv = np.linalg.inv(Q)

    iteration = 0
    while iteration < ITER_LIMIT:
        last = next
        next = Qinv.dot(Q_minus_A.dot(last) + b)

        # check the convergence criterion
        if norm(A.dot(next) - b) / b_norm < 10 ** -6:
            print("\n   done @ iteration", iteration + 1)
            break

        if iteration % 10 == 0 or iteration > ITER_LIMIT - 10:
            progress_bar(iteration, ITER_LIMIT - 1)

        iteration += 1
        if iteration == ITER_LIMIT:
            print("   exhausted the iteration limit")

    return next

for gamma in [5, 2, 0.5]:
    print("gamma", gamma)
    for method in [jacobi, gauss_seidel]:
        print("method", method.__name__)
        (A, b) = generate_input(gamma)
        iteratively(method, A, b)
