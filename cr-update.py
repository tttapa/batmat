# %%

import numpy as np
import numpy.linalg as la

np.set_printoptions(threshold=np.inf, linewidth=500)


def cyclic_indices(lgn, lgv=0):
    if lgn != lgv:
        for i in range(1 << (lgn - 1)):
            yield 2 * i + 1
        for l in range(1, lgn - lgv):
            offset = 1 << l
            stride = 2 * offset
            for i in range(offset, 1 << lgn, stride):
                yield i
    for i in range(0, 1 << lgn, 1 << (lgn - lgv)):
        yield i


def cyclic_indices_lim(N, lgp, lgv=0):
    assert N >= (1 << lgp)
    assert N % (1 << lgp) == 0
    stride = N >> lgp
    for k in range(1 << lgp):
        for i in range(stride - 2):
            yield 1 + i + k * stride
    if stride > 1:
        for k in range(1 << lgp):
            yield stride - 1 + k * stride
    for k in cyclic_indices(lgp, lgv):
        yield k * stride


def updowndate(S, A, L):
    l, n = L.shape
    lA, m = A.shape
    assert l >= n
    assert lA == l
    T = np.zeros((0, 0))
    B = np.zeros((0, m))
    L̃ = np.zeros((l, 0))
    for k in range(n):
        a = A[k, :]
        Aʹ = A[k + 1 :, :]
        λ = L[k, k]
        l = L[k + 1 :, k]
        α2 = np.dot(a, S * a)
        λ̃ = np.copysign(np.sqrt(λ**2 + α2), λ)
        β = λ + λ̃
        γ = 2 * β**2 / (α2 + β**2)
        b = a / β
        w = γ * (l + Aʹ @ (S * b))
        l̃ = w - l
        Aʹ -= np.outer(w, b)
        τ = np.array([[1 / γ]])
        T = np.block([[T, B @ np.vstack(S * b)], [np.zeros((1, k)), τ]])
        B = np.vstack([B, np.hstack(b)])
        l̃_full = np.concatenate([np.zeros(k), np.array([λ̃]), l̃])
        L̃ = np.hstack([L̃, np.vstack(l̃_full)])
    A[:n, :] = B
    L[:] = L̃
    return T


def updown_apply(S, T, B, A, L):
    Wʹ = L + A @ (S[:, None] * B.T)
    W = Wʹ @ la.inv(T)
    A[:] -= W @ B
    L[:] = W - L


# fmt: off
assert list(cyclic_indices_lim(16, 4, 0)) == [1, 3, 5, 7, 9, 11, 13, 15, 2, 6, 10, 14, 4, 12, 8, 0]
assert list(cyclic_indices_lim(16, 4, 1)) == [1, 3, 5, 7, 9, 11, 13, 15, 2, 6, 10, 14, 4, 12, 0, 8]
assert list(cyclic_indices_lim(16, 4, 2)) == [1, 3, 5, 7, 9, 11, 13, 15, 2, 6, 10, 14, 0, 4, 8, 12]
assert list(cyclic_indices_lim(16, 4, 3)) == [1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14]
assert list(cyclic_indices_lim(16, 4, 4)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
assert list(cyclic_indices_lim(8, 3, 2)) == [1, 3, 5, 7, 0, 2, 4, 6]
assert list(cyclic_indices_lim(24, 3, 2)) == [1, 4, 7, 10, 13, 16, 19, 22, 2, 5, 8, 11, 14, 17, 20, 23, 3, 9, 15, 21, 0, 6, 12, 18]
# fmt: on

# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_sparse(A_sparse):
    A_dense = A_sparse.toarray() if hasattr(A_sparse, "toarray") else A_sparse

    A_plot = abs(A_dense)
    A_plot[A_plot == 0] = np.nan
    nrm = LogNorm(vmin=np.nanmin(A_plot), vmax=np.nanmax(A_plot), clip=False)
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


# %%


def get_level(i: int) -> int:
    assert i > 0
    return (i & -i).bit_length() - 1


def get_index_in_level(i: int) -> int:
    if i == 0:
        return 0
    l = get_level(i)
    return i >> (l + 1)


def get_linear_batch_offset(biA: int, lP: int, lvl: int) -> int:
    levA = get_level(biA) if biA > 0 else lP
    levP = lP - lvl
    if levA >= levP:
        return (((1 << levP) - 1) << (lP - levP)) + (biA >> levP)
    return (((1 << levA) - 1) << (lP - levA)) + get_index_in_level(biA)


# %%


def build_cr_factor(D, Y, U):
    N, nx, _ = D.shape
    lP = int(round(np.log2(N)))
    lvl = 0
    A = np.zeros((N * nx, N * nx))

    def cyclic_block(s: int, i: int, offset: int) -> int:
        sY = nx * get_linear_batch_offset(i + offset, lP, lvl)
        sU = nx * get_linear_batch_offset(i - offset, lP, lvl)
        bi = i % (1 << (lP - lvl))
        vi = i // (1 << (lP - lvl))
        A[s : s + nx, s : s + nx] = D[bi]
        if i + offset < (1 << lP):
            A[sY : sY + nx, s : s + nx] = Y[bi]
        A[sU : sU + nx, s : s + nx] = U[bi]

    def cyclic_block_final(s: int, i: int, offset: int) -> int:
        sY = nx * get_linear_batch_offset(i + offset, lP, lvl)
        bi = i % (1 << (lP - lvl))
        vi = i // (1 << (lP - lvl))
        A[s : s + nx, s : s + nx] = D[bi]
        if i + offset < (1 << lP):
            A[sY : sY + nx, s : s + nx] = Y[bi]

    s = 0
    if lP != lvl:
        for i in range(1 << (lP - 1)):
            cyclic_block(s, 2 * i + 1, 1)
            s += nx
        for l in range(1, lP - lvl):
            offset = 1 << l
            stride = offset << 1
            for i in range(offset, 1 << lP, stride):
                cyclic_block(s, i, offset)
                s += nx
    for i in range(0, 1 << lP, 1 << (lP - lvl)):
        cyclic_block_final(s, i, 1 << (lP - lvl))
        s += nx

    return A


def build_cr_update(D, S):
    # Create a lower bidiagonal matrix and perform a CR permutation of the rows
    #
    #   [ D0             ]
    # P [ S0  D1         ]
    #   [     S1  D2     ]
    #   [         S2  D3 ]
    N, nx, ny = D.shape
    lP = int(round(np.log2(N)))
    A = np.zeros((N * nx, N * ny))
    for j in range(N):
        A[j * nx : (j + 1) * nx, j * ny : (j + 1) * ny] = D[j]
        if j + 1 < N:
            A[(j + 1) * nx : (j + 2) * nx, j * ny : (j + 1) * ny] = S[j]
    iotas = np.tile(np.arange(nx), N)
    sel = np.repeat(np.array(list(cyclic_indices(lP))) * nx, nx) + iotas
    return A[sel, :]


rng = np.random.default_rng(seed=123)
lP = 4
N = 1 << lP
nx = 2
ny = 3
LD = np.tril(rng.uniform(-1, 1, (N, nx, nx)))
Y = rng.uniform(-1, 1, (N - 1, nx, nx))
U = rng.uniform(-1, 1, (N, nx, nx))
L = build_cr_factor(LD, Y, U)
plot_sparse(L @ L.T)

# %%

A1 = rng.uniform(-1, 1, (N, nx, ny))
A2 = 1e-3 * rng.uniform(-1, 1, (N, nx, ny))

A1 = 100 + np.kron(np.arange(0, N), np.ones((ny, nx))).reshape((ny, nx, N), order="F").T
A2 = 200 + np.kron(np.arange(0, N - 1), np.ones((ny, nx))).reshape((ny, nx, N - 1), order="F").T

A = build_cr_update(A1, A2)
plot_sparse(A)
A

# %%

S = 2**rng.uniform(-1, 1, (N * ny))
T = updowndate(S, A_upd := np.copy(A), L_upd := np.copy(L))
plot_sparse(A_upd)

# %%

LD_upd = np.copy(LD)
U_upd = np.copy(U)
Y_upd = np.copy(Y)

def build_cr_update_workspace(A1, A2):
    N, nx, ny = A1.shape
    assert N > 2
    W = np.zeros((4, nx, N * ny))
    for i in range(N):
        if i & 1:
            W[0, :, i * ny : (i + 1) * ny] = A1[i]
        else:
            W[0, :, i * ny : (i + 1) * ny] = A2[i]
        if (i & 3) == 0:
            W[2, :, i * ny : (i + 1) * ny] = A1[i]
        elif (i & 3) == 1:
            W[1, :, i * ny : (i + 1) * ny] = A2[i]
        elif (i & 3) == 2:
            W[1, :, i * ny : (i + 1) * ny] = A1[i]
        elif i + 1 < N:
            W[2, :, i * ny : (i + 1) * ny] = A2[i]
    return W

def build_cr_update_workspace(A1, A2):
    # Builds the workspace used during a factorization update of a CR
    # factorization:
    #
    # [ S0 D1 S2 D3 S4 D5 S6 D7 ]
    # [  · S1 D2  ·  · S5 D6  · ]
    # [ D0  ·  · S3 D4  ·  · S7 ]
    # [  ·  ·  ·  ·  ·  ·  ·  · ]
    N, nx, ny = A1.shape
    assert N > 2
    W = np.zeros((4, nx, N * ny))
    for i in range(N):
        match i & 3:
            case 0: rD, rS = 2, 0
            case 1: rD, rS = 0, 1
            case 2: rD, rS = 1, 0
            case 3: rD, rS = 0, 2
        W[rD, :, i * ny : (i + 1) * ny] = A1[i]
        if i + 1 < N:
            W[rS, :, i * ny : (i + 1) * ny] = A2[i]
    return W

W = build_cr_update_workspace(A1, A2)
print(W[:3, ::nx, ::ny])

def update_level(l):
    for i in range(N >> (l + 1)):
        c0 = (ny * i) << (l + 1)
        c1 = (ny * (i + 1)) << (l + 1)
        bi = (i << (l + 1)) + (1 << l)
        T = updowndate(S[c0:c1], W[l % 4, :, c0:c1], LD_upd[bi])

        match i & 3:
            case 0 | 3: dst = W[(l + 3) % 4, :, c0:c1]; dst[:] = W[(l + 2) % 4, :, c0:c1]; W[(l + 2) % 4, :, c0:c1] = 0
            case 1 | 2: dst = W[(l + 2) % 4, :, c0:c1]

        if i & 1:
            updown_apply(S[c0:c1], T, W[(l + 0) % 4, :, c0:c1], W[(l + 1) % 4, :, c0:c1], U_upd[bi])
            if bi + (1 << l) < N:
                updown_apply(S[c0:c1], T, W[(l + 0) % 4, :, c0:c1], dst, Y_upd[bi])
        else:
            updown_apply(S[c0:c1], T, W[(l + 0) % 4, :, c0:c1], W[(l + 1) % 4, :, c0:c1], Y_upd[bi])
            updown_apply(S[c0:c1], T, W[(l + 0) % 4, :, c0:c1], dst, U_upd[bi])
        W[(l + 0) % 4, :, c0:c1] = 0

for i in range(lP):
    update_level(i)
updowndate(S, W[(lP + 2) % 4], LD_upd[0])


# %%

plot_sparse(build_cr_factor(LD_upd, Y_upd, U_upd) - L_upd)

# %%

plt.show()

# %%
