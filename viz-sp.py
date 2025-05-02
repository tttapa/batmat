import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import coo_matrix, diags


def load_mat(f, make_sym=False) -> coo_matrix:
    data = pd.read_csv(
        f,
        header=None,
        names=["r", "c", "x"],
        dtype={"r": np.int32, "c": np.int32, "x": np.float64},
    )

    rows = data["r"].values
    cols = data["c"].values
    values = data["x"].values
    if make_sym:
        rows, cols, values = (
            np.concatenate((rows[rows >= cols], cols[rows > cols])),
            np.concatenate((cols[rows >= cols], rows[rows > cols])),
            np.concatenate((values[rows >= cols], values[rows > cols])),
        )

    n = max(rows.max(), cols.max()) + 1
    A_sparse = coo_matrix((values, (rows, cols)), shape=(n, n))
    return A_sparse


A_sparse = load_mat("sparse.csv", make_sym=True)
L_sparse = load_mat("sparse_factor.csv")
D_sparse = load_mat("sparse_diag.csv")
print(A_sparse.shape)
print(L_sparse.shape)
print(D_sparse.shape)

n1 = 9280
n2 = 40 * 4

A = A_sparse.toarray()
L = L_sparse.toarray()
D = D_sparse.toarray()
S = -A[n1:, n1:] + L[n1:, :n1] @ D[:n1, :n1] @ L[n1:, :n1].T
L[-n2:, -n2:] = np.linalg.cholesky(L[-n2:, -n2:])
SLL = L[n1:, n1:] @ L[n1:, n1:].T

LS = np.linalg.cholesky(S)


def plot_sparse(A_sparse):
    A_dense = A_sparse.toarray() if hasattr(A_sparse, "toarray") else A_sparse

    A_plot = abs(A_dense)
    A_plot[A_plot == 0] = np.nan
    nrm = LogNorm(vmin=np.nanmin(A_plot), vmax=np.nanmax(A_plot), clip=False)
    A_plot[np.logical_not(np.isfinite(A_dense))] = nrm.vmin / 2
    print(np.nanmin(A_plot))

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")  # Show zeros (NaNs) as white
    cmap.set_under(color="red")  # for original NaNs (set as -1)

    plt.imshow(A_plot, norm=nrm, cmap=cmap)
    plt.colorbar(label="|x| (log scale)")
    plt.title("Sparse Matrix Visualization (Log Scale, Abs Values)")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()

plot_sparse(LS - L[n1:, n1:])
plot_sparse(S - SLL)
# plot_sparse(A_sparse)
# plot_sparse(L_sparse)
# # plot_sparse(D_sparse)
# LL_sparse = L_sparse @ D_sparse @ L_sparse.T
# plot_sparse(A_sparse - LL_sparse)

plt.show()
