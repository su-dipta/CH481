import numpy as np
import time

# ------------------------------------------------------------
# 1. Define the problem (fake "Hamiltonian")
# ------------------------------------------------------------

n = 50             # Dimension of the matrix
tol = 1e-8             # Convergence tolerance
mmax = n // 2          # Maximum subspace size
sparsity = 50      # Controls diagonal dominance

# Build a symmetric, diagonally dominant test matrix
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = i + 1
A = A + sparsity * np.random.randn(n, n)
A = (A.T + A) / 2

# ------------------------------------------------------------
# 2. Davidson parameters
# ------------------------------------------------------------

k = 3                  # block size
eig = 2                # number of eigenvalues to solve
t = np.eye(n, k)       # initial guess vectors
V = np.zeros((n, n))   # space to hold subspace basis
I = np.eye(n)          # identity matrix same size as A
# ------------------------------------------------------------
# 3. Davidson iterative loop
# ------------------------------------------------------------

print("Starting Davidson iterations...")
start_davidson = time.time()

for m in range(k, mmax, k):
    print("m =", m)

    # ---- Initial step ----
    if m <= k:
        for j in range(k):
            V[:, j] = t[:, j] / np.linalg.norm(t[:, j])
            print(V[:,j])
        theta_old = np.ones(eig) * 1e6
    else:
        theta_old = theta[:eig].copy()
    # ---- Orthonormalize current subspace ----
    V[:, :m], R = np.linalg.qr(V[:, :m])

    # ---- Build small projected matrix ----
    T = np.dot(V[:, :m].T, np.dot(A, V[:, :m]))

    # ---- Solve eigenproblem in small space ----
    THETA, S = np.linalg.eigh(T)
    idx = np.argsort(THETA)
    print(idx)
    theta = THETA[idx]
    s = S[:, idx]
    print(S.shape)

    # ---- Compute residuals and Davidson corrections ----
    residuals = []
    for j in range(k):
        xj = np.dot(V[:, :m], s[:, j])       # Ritz vector in full space
        rj = np.dot(A, xj) - theta[j] * xj   # residual
        residuals.append(np.linalg.norm(rj))

        # Davidson correction using diagonal preconditioner
        q = rj / (theta[j] - np.diag(A))
        print("q shape ", q)
        q[np.isnan(q)] = 0.0
        q[np.isinf(q)] = 0.0
        if m + j < n:
            V[:, m + j] = q
        #print("q ", q.shape)
    print("V ", V)
    print(theta_old.shape, theta.shape)
    residuals = np.array(residuals[:eig])
    delta = np.linalg.norm(theta[:eig] - theta_old)

    # ---- Print progress ----
    print(f"Iter {m//k:3d} | Subspace dim = {m:4d} | "
          f"E = {theta[:eig]} | "
          f"delta E = {delta:.2e} | "
          f"Residual norms = {residuals}")

    # ---- Convergence check ----
    if delta < tol and np.all(residuals < tol):
        print("Converged!\n")
        break

end_davidson = time.time()
print(f"Davidson converged eigenvalues: {theta[:eig]}")
print(f"Davidson time: {end_davidson - start_davidson:.3f} s")

# ------------------------------------------------------------
# 4. Full diagonalization for comparison
# ------------------------------------------------------------

print("\nRunning NumPy full diagonalization for comparison...")
start_numpy = time.time()

E, Vec = np.linalg.eigh(A)
E = np.sort(E)

end_numpy = time.time()

print(f"Numpy eigenvalues: {E[:eig]}")
print(f"Numpy time: {end_numpy - start_numpy:.3f} s")

