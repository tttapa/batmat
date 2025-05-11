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
D_sparse = load_mat("sparse_diag.csv")
print(A_sparse.shape)
print(L_sparse.shape)
print(D_sparse.shape)

sol_ref = spla.spsolve(A_sparse, rhs)
# print(sol)
# print(sol_ref)

n1 = 5376
n2 = 6 * 4

A = A_sparse.toarray()
L = L_sparse.toarray()
D = D_sparse.toarray()
# S = -A[n1:, n1:] + L[n1:, :n1] @ D[:n1, :n1] @ L[n1:, :n1].T
# # L[-n2:, -n2:] = np.linalg.cholesky(L[-n2:, -n2:])
# SLL = L[n1:, n1:] @ L[n1:, n1:].T

# LS = np.linalg.cholesky(S)


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

LL_sparse = L_sparse @ D_sparse @ L_sparse.T
if 0:
    plot_sparse(LS - L[n1:, n1:])
    plot_sparse(S - SLL)
    # plot_sparse(A_sparse)
    # plot_sparse(L_sparse)
    # # plot_sparse(D_sparse)
    plot_sparse(A_sparse - LL_sparse)
    plot_sparse(L_sparse)
plt.figure()
plt.semilogy(abs(sol_ref - sol), ".-")
plt.semilogy(abs(sol_ref + sol), ".-")
plt.axvline(n1, c='k', lw=0.5)

b = rhs[6:9]
x = sol[3:6]
λ = sol[6:9]
λ_ref = sol_ref[6:9]
LQ0 = L[3:6, 3:6]
# print(LQ0.T @ b + x)
# print(λ)
# print(λ_ref)


# rhs2_ref = rhs[n1:] - L[n1:, :n1] @ D[:n1, :n1] @ sol_ref[:n1]
# print(rhs2_ref)
# print(sol[n1:])

plt.show()

# k= 0  di= 0  k_next=47  di_next= 1
# k=47  di= 1  k_next=46  di_next= 2
# k= 3  di= 3  k_next= 2  di_next= 4
# k= 2  di= 4  k_next= 1  di_next= 5
# k= 6  di= 6  k_next= 5  di_next= 7
# k= 5  di= 7  k_next= 4  di_next= 8
# k= 9  di= 9  k_next= 8  di_next=10
# k= 8  di=10  k_next= 7  di_next=11

print(rhs)

b0 = np.array(
[+5.95711771768213882e-01, +4.89537141268501896e-01, -3.96407866702587874e-01])
B̂0 = np.array(
[[+4.19668970264160182e-02, +1.28166455458074063e-01, +2.19552962246310795e-01],
 [+1.72173987055492195e-01, -1.85802234350153722e-01, -2.81749443138696709e-01],
 [+4.00840057803141692e-01, -8.37227741804769804e-03, +1.09594562556698513e-01]])
Â0 = np.array(
[[-5.17216759851258367e-01, -1.86237212779923689e+00, -4.12535246951326817e-01],
 [-5.71451965742796997e-01, -3.09581670087909766e-02, +1.03627794559310438e+00],
 [+1.11024021317469224e+00, -1.08868520958233891e-01, -7.74749117161982609e-02]])
LQ̂0 = np.array(
[[+3.51685690320111322e+00, +0.00000000000000000e+00, +0.00000000000000000e+00],
 [+5.12892798770628530e-02, +3.24640906556677322e+00, +0.00000000000000000e+00],
 [+1.27938753475402817e-01, +8.73880213865915884e-02, +3.44675795249223915e+00]])
l0 = np.array(
[-9.81082746141579976e-02, -1.70418797642961611e-01, +1.92766598058553729e-01])
q̃0 = np.array(
[-1.49946794847438170e-01, +1.61001776731961810e-01, +1.89570118839885504e-01])
b̃1 = np.array(
[+1.37596351530874550e+00, +1.03118608645483478e+00, +9.03017823895189253e-01])
B̂1 = np.array(
[[+3.73329615782366842e-01, -9.76493368833989683e-02, +3.62270298926959181e-02],
 [-9.48763688203428518e-02, +5.34775662197052512e-02, -2.72087704267392760e-01],
 [-1.21205035913814319e-01, -5.52303290355570520e-02, +2.55904068089249925e-01]])
Â1 = np.array(
[[-6.99311557278420537e-01, -5.04987429723489090e-01, +1.22558431719541971e+00],
 [-1.36007124237195964e-01, -1.36882236328856710e-01, +6.77478366031903478e-01],
 [-3.66441820598629037e-02, +9.92891694393945312e-02, +2.49180697327979150e-01]])
LQ̂1 = np.array(
[[+3.70295723438764846e+00, +0.00000000000000000e+00, +0.00000000000000000e+00],
 [+1.08988087260288680e+00, +3.45385943449279242e+00, +0.00000000000000000e+00],
 [-5.65330027088188469e-01, -8.09900744298008735e-01, +5.29793479442784587e+00]])
l1 = np.array(
[-1.60451533857507278e+00, +1.31514135876898447e-01, +3.15975615295748968e-01])
q̃1 = np.array(
[+4.69248694963482371e-01, +1.08577442083165263e-01, +1.88896324547070560e-01])
b̃2 = np.array(
[+3.21280979372416686e+00, +2.14039963568001612e+00, +3.08120140530938125e+00])
B̂2 = np.array(
[[+1.16495182233355365e-01, +1.70859308749017696e-01, -7.21917643741843834e-02],
 [+8.16084875801092319e-02, +6.11048953054374525e-02, -2.62000359539037121e-03],
 [+1.18396675942952761e-02, +2.69601999404171788e-02, +2.54486991409179110e-02]])
Â2 = np.array(
[[-1.12254712034637591e-01, -9.77972218895173800e-02, -6.56570537919493880e-02],
 [-3.74607192871828990e-02, -5.47752753266720696e-02, -1.55787552343215224e-03],
 [-1.30182323800629249e-02, -2.82239172959737501e-02, +1.72825098727588354e-02]])
LQ̂2 = np.array(
[[+4.28451303248160009e+00, +0.00000000000000000e+00, +0.00000000000000000e+00],
 [+2.31158912252308779e+00, +5.91351243794090209e+00, +0.00000000000000000e+00],
 [+3.43395167296326731e+00, +1.01423954088954971e+00, +6.62233504084451852e+00]])
l2 = np.array(
[+1.75045569628213959e+00, -5.99011562550657017e-01, +3.03554146771570821e+00])
q̃2 = np.array(
[+8.63817997879478994e-01, -1.67993101032984882e+00, +2.43671913210540314e+00])
b̂0 = np.array(
[+4.75620351140193742e-01, +4.54924176342237141e-01, -1.98642587399278536e-01])

LÂ0 = Â0 @ la.inv(LQ̂0.T)
LÂ1 = Â1 @ la.inv(LQ̂1.T)
LÂ2 = Â2
del Â2

b̂0_check = b0 - B̂0 @ l0 - LÂ0 @ q̃0 + LÂ0 @ b̃1 \
              - B̂1 @ l1 - LÂ1 @ q̃1 + LÂ1 @ b̃2 \
              - B̂2 @ l2 - LÂ2 @ q̃2
print(b̂0)
print(b̂0_check)
