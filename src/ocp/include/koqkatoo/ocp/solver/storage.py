# This file automatically generates one big batched matrix and divides it up
# into all the different workspaces we need.
# It outputs C++ code that goes into <koqkatoo/ocp/solver/storage.hpp>.

import sympy as sy
import itertools as it

nx, nu, ny = "dim.nx", "dim.nu", "dim.ny"
nxu = "(dim.nx + dim.nu)"
nxux = "(dim.nx + dim.nx + dim.nu)"

nx, nu, ny = sy.symbols("nx nu ny")
nxu = nx + nu
nxux = nxu + nx
stage_matrices = {
    "CD": (ny, nxu),
    "HAB": (nxu + nx, nxu),
    "LHV": (nxux, nxu),
    "λ1": (nx, 1),
    "mFx": (nx, 1),
    "Wᵀ": (nxu, nx),
    "WWᵀVWᵀ": (2 * nx, nx),
    "VVᵀ": (nx, nx),
    "Z": (nxu, ny),  # assuming ny >= ny_N
    "Z1": (nxu + 2 * nx, ny),  # assuming ny >= ny_N
    "Σ_sgn": (ny, 1),  # assuming ny >= ny_N
    "Σ_ud": (ny, 1),  # assuming ny >= ny_N
    "Lupd": (ny, ny),  # assuming ny >= ny_N
    "Wupd": (nx, ny),  # assuming ny >= ny_N
    "Vupd": (nx, ny),  # assuming ny >= ny_N
}

stage_matrices_sizes = {
    k: sy.Mul(*rc, evaluate=False) for k, rc in stage_matrices.items()
}
offsets = [0] + list(it.accumulate(stage_matrices_sizes.values()))
stage_matrices_offsets = {k: offsets[i] for i, k in enumerate(stage_matrices)}
stage_matrices_total_size = offsets[-1]


def pow_to_mul(expr):
    """
    Convert integer powers in an expression to Muls, like a**2 => a*a.
    Handles nested cases like 2*x**2 => 2*x*x.
    """

    def expand_powers(expr):
        # Recursively process the arguments if the expression is not an atomic type
        if expr.is_Atom:
            return expr
        elif expr.is_Pow and expr.exp.is_integer and expr.exp > 0:
            # Replace power with multiplication
            return sy.Mul(*[expand_powers(expr.base)] * expr.exp, evaluate=False)
        elif expr.is_Mul or expr.is_Add:
            # Process arguments of the addition or multiplication
            return expr.func(*[expand_powers(arg) for arg in expr.args], evaluate=False)
        else:
            # For other types, return as is
            return expr

    # Expand the expression initially to simplify nested terms
    expr = sy.expand(expr)
    return expand_powers(expr)


print(
    f"""    // clang-format off
    real_matrix stagewise_storage = [this] {{
        auto [N, nx, nu, ny, ny_N] = dim;
        return real_matrix{{{{
            .depth = N + 1,
            .rows = {stage_matrices_total_size},
            .cols = 1,
        }}}};
    }}();

    #define KQT_PRECOMPUTE_OFFSET(name, value)                                 \\
        const index_t name ## _offset = [this] {{                               \\
            [[maybe_unused]] auto [N, nx, nu, ny, ny_N] = dim;                 \\
            return value;                                                      \\
        }}()
"""
)

for k, (r, c) in stage_matrices.items():
    off = stage_matrices_offsets[k]
    try:
        print(f"""    static constexpr index_t {k}_offset = {int(off)};""")
    except TypeError as e:
        print(f"""    KQT_PRECOMPUTE_OFFSET({k}, {pow_to_mul(off)});""")

for k, (r, c) in stage_matrices.items():
    size = stage_matrices_sizes[k]
    print(
        f"""
    mut_real_view {k}() {{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows({k}_offset, {size}).reshaped({r}, {c});
    }}"""
    )

print(
    """
    #undef KQT_PRECOMPUTE_OFFSET
    // clang-format on
"""
)
