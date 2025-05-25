import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import coo_matrix, diags
import scipy.sparse as spa
import scipy.sparse.linalg as spla


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

rhs = np.loadtxt("rhs.csv")
sol = np.loadtxt("sol.csv")
A_sparse = load_mat("sparse.csv", make_sym=True)
L_sparse = load_mat("sparse_factor.csv")
L_ref_sparse = load_mat("sparse_refactor.csv")
D_sparse = load_mat("sparse_diag.csv")
print(A_sparse.shape)

sol_ref = spla.spsolve(A_sparse, rhs)

nx, nu = 2, 1
n2 = nx * 4


def plot_sparse(A_sparse):
    A_dense = A_sparse.toarray() if hasattr(A_sparse, "toarray") else A_sparse

    A_plot = abs(A_dense)
    A_plot[A_plot < 1e-50] = np.nan
    nrm = LogNorm(vmin=np.nanmin(A_plot) or 1e-50, vmax=min(np.nanmax(A_plot), 1e50) or 1e-49, clip=False)
    A_plot[np.logical_not(np.isfinite(A_dense))] = nrm.vmin / 2

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

LL_sparse = L_sparse @ D_sparse @ L_sparse.T
if 1:
    # plot_sparse(A_sparse)
    # plot_sparse(L_sparse)
    # # plot_sparse(D_sparse)
    # err = (A_sparse - LL_sparse).toarray()
    eps = 1e-20
    err = (L_ref_sparse - L_sparse).toarray()
    sparsity = np.logical_or(A_sparse.toarray() != 0, L_sparse.toarray() != 0)
    sparsity = L_ref_sparse.toarray() != 0
    err[np.logical_and(sparsity, abs(err) < eps)] = eps
    plot_sparse(err)
    plot_sparse(L_sparse)
    # plot_sparse(L_sparse.tocsc()[:nu + nx, :nu + nx] - la.cholesky(A_sparse.tocsc()[:nu + nx, :nu + nx].toarray()))
plt.figure()
plt.semilogy(abs(sol_ref - sol), ".-")
plt.semilogy(abs(sol_ref + sol), ".-")
# plt.axvline(n1, c='k', lw=0.5)

plt.show()
