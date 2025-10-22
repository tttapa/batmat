import contextlib
import json
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\renewcommand{\sfdefault}{phv}\renewcommand{\rmdefault}{ptm}",
        "font.family": "ptm",
        "font.size": 14,
        "figure.titlesize": 16,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.25,
    }
)

filename = Path("benchmarks/results/benchmark-gemm.json")
with contextlib.suppress(IndexError):
    filename = Path(sys.argv[1])

# Load the JSON data
with open(filename) as f:
    data = json.load(f)

blasfeo_data = None
with contextlib.suppress(FileNotFoundError):
    blasfeo_filename = "blasfeo-" + filename.stem.replace("avx512", "avx2")
    blasfeo_filename = filename.with_stem(blasfeo_filename)
    blasfeo_filename = filename.with_stem("blasfeo-" + filename.stem)
    with open(blasfeo_filename) as f:
        blasfeo_data = json.load(f)

# Create a DataFrame from the benchmarks
stat = "median"
metric = "real_time"
gflops_max = {
    "Nutella": 40,
    "blacksad.esat.kuleuven.be": 20,
    "ketelbuik.esat.kuleuven.be": 39.2,
}.get(data["context"]["host_name"])

isa = data["context"]["arch"]
df = pd.DataFrame(data["benchmarks"] + (blasfeo_data or {}).get("benchmarks", []))
# Remove aggregates computed by gbench
df = df[df["aggregate_name"].isna()]
# Extract the final part of the name after the last slash for the x-axis
df["version"] = df["run_name"].apply(lambda x: int(x.split("/", 2)[1]))
df.sort_values(by="version", inplace=True, kind="stable")
df["GFLOPS"] = 1e9 * df["GFLOP count"] / df[metric]
# Group the data and aggregate different runs of the same run
df_runs = df[["run_name", "real_time", "cpu_time", "GFLOPS"]].groupby("run_name")
df = df_runs.aggregate([stat, "min", "max"])
df["run_name"] = df.index
df["version"] = df["run_name"].apply(lambda x: int(x.split("/", 2)[1]))
df.sort_values(by="version", inplace=True, kind="stable")
df["full name"] = df["run_name"]
df["run_name"] = df["run_name"].apply(lambda x: x.split("/", 2)[0])


def parse_run_name(name):
    """Split into function name and list of template args."""
    m = re.match(r"(\w+)<(.*)>", name)
    if not m:
        return name, ()
    func, args_str = m.groups()
    args = tuple(a.strip() for a in args_str.split(","))
    return func, args

df[["func", "args"]] = df["run_name"].apply(parse_run_name).apply(pd.Series)

def generate_ref_name_from_parts(func, args, existing_names):
    args = ("scalar",) + args[1:]
    for i in range(len(args), 0, -1):  # Try progressively shorter templates
        candidate = f"{func}<{', '.join(args[:i])}>"
        if candidate in existing_names:
            return candidate
    return None

def add_reference_columns(df: pd.DataFrame, metrics):
    # Generate reference names for each run
    existing_names = set(df["run_name"])
    df["ref_name"] = df.apply(
        lambda r: generate_ref_name_from_parts(r["func"].item(), r["args"].item(), existing_names),
        axis=1
    )

    # Build the "reference dataframe" (scalar_abi runs)
    ref_df = df[["run_name", "version"] + metrics].copy()
    ref_df = ref_df.rename(columns={"run_name": "ref_name"})

    # For MultiIndex columns, rename top-level only
    if isinstance(ref_df.columns, pd.MultiIndex):
        ref_df.columns = pd.MultiIndex.from_tuples([
            (f"{a}_ref" if i == 0 and a in metrics else a, b)
            for i, (a, b) in enumerate(ref_df.columns)
        ])
    else:
        ref_df = ref_df.rename(columns={m: f"{m}_ref" for m in metrics})

    # Merge on (ref_name, version)
    df = df.merge(ref_df, on=["ref_name", "version"], how="left", suffixes=("", "_ref"))
    return df.drop(columns="ref_name")

def benchmark_label(func_name: str, args: tuple[str]) -> str:
    # Map ABI
    impl = "hyhound" if func_name == "hyh" else "MKL"
    if args[0] == "scalar":
        abi_label = f"{impl} AVX2" if isa == "avx2" else f"{impl} AVX-512"
    elif args[0] == "blasfeo":
        abi_label = f"BLASFEO AVX2" if isa == "avx2" else f"BLASFEO AVX-512"
    elif args[0] == "simd4":
        abi_label = "batmat AVX2 (4)" if isa == "avx2" else "batmat AVX-512 (4)"
    elif args[0] == "simd8":
        abi_label = "batmat AVX2 (8)" if isa == "avx2" else "batmat AVX-512 (8)"
    else:
        abi_label = "unknown"

    # gemm: args = (Abi, OA, OB, PA?, PB?, Tiling?)
    if func_name == "gemm":
        tiling_label = ""
        if len(args) > 5 and args[5].lower() == "false":
            tiling_label = " (no tiling)"
            return None
        elif len(args) > 4:
            if args[3].lower() == args[4].lower() == "never":
                tiling_label = " (no packing)"
                return None
            elif args[3].lower() == args[4].lower() == "always":
                tiling_label = " (full packing)"
                return None
            elif args[3].lower() == args[4].lower() == "transpose":
                tiling_label = " (transpose packing)"
                return None
            elif args[3].lower() == "always" and args[4].lower() == "transpose":
                tiling_label = " ($A$ full, $B$ transpose packing)"
                return None
            elif args[3].lower() == "transpose" and args[4].lower() == "always":
                tiling_label = " ($A$ transpose, $B$ full packing)"
                tiling_label = " (with packing)"
        return f"{abi_label}{tiling_label}"
    return abi_label

def benchmark_color(func_name: str, args: tuple[str], label: str) -> str:
    if args[0] == "blasfeo":
        return "tab:pink"
    elif args[0] == "simd4":
        if "no tiling" in label:
            return "tab:blue"
        return "tab:green"
    elif args[0] == "simd8":
        if "no tiling" in label:
            return "tab:orange"
        return "tab:red"
    return "tab:purple"

# Plot the data
df["label"] = df.apply(lambda r: benchmark_label(r["func"].item(), r["args"].item()), axis=1)
df = df[df["label"].notna()]
df["color"] = df.apply(lambda r: benchmark_color(r["func"].item(), r["args"].item(), r["label"].item()), axis=1)
df: pd.DataFrame = add_reference_columns(df, ["GFLOPS", metric])
functions = df["run_name"].unique()
title = filename.stem.replace("benchmark-", "").replace("-", " ").upper()
title = {
    "GEMM": "Matrix Multiplication (\\texttt{gemm})",
    "POTRF": "Cholesky Factorization (\\texttt{potrf})",
    "SYRK": "Symmetric Rank-$k$ Update (\\texttt{syrk})",
    "SYRK POTRF": "Merged Symmetric Rank-$k$ Update and Cholesky Factorization (\\texttt{syrk+potrf})",
    "TRTRI": "Triangular Inverse (\\texttt{trtri})",
    "TRMM": "Triangular Matrix Multiplication (\\texttt{trmm})",
    "TRSM": "Triangular Matrix Solve (\\texttt{trsm})",
    "HYH": "Cholesky Factorization Update (\\texttt{hyhound})",
}.get(title, title)

separate_figs = {
    "gemm": {
        ("RowMajor", "ColMajor"): "\\textsc{gemm} $D_c = A_r B_c$",
        ("ColMajor", "ColMajor"): "\\textsc{gemm} $D_c = A_c B_c$",
        ("RowMajor", "RowMajor"): "\\textsc{gemm} $D_c = A_r B_r$",
        ("ColMajor", "RowMajor"): "\\textsc{gemm} $D_c = A_c B_r$",
    },
    "trmm": {
        ("Right", "RowMajor", "ColMajor"): "\\textsc{trmm} $D_r = A_r L_c$",
        ("Right", "ColMajor", "ColMajor"): "\\textsc{trmm} $D_c = A_c L_c$",
        ("Right", "RowMajor", "RowMajor"): "\\textsc{trmm} $D_r = A_r L_r$",
        ("Right", "ColMajor", "RowMajor"): "\\textsc{trmm} $D_c = A_c L_r$",
    },
    "trsm": {
        ("Right", "RowMajor", "ColMajor"): "\\textsc{trsm} $D_r = A_r L_c^{-1}$",
        ("Right", "ColMajor", "ColMajor"): "\\textsc{trsm} $D_c = A_c L_c^{-1}$",
        ("Right", "RowMajor", "RowMajor"): "\\textsc{trsm} $D_r = A_r L_r^{-1}$",
        ("Right", "ColMajor", "RowMajor"): "\\textsc{trsm} $D_c = A_c L_r^{-1}$",
    },
    "syrk": {
        ("RowMajor", "ColMajor"): "\\textsc{syrk} $D_c = C_c + A_r A_r^\\top$",
        ("ColMajor", "ColMajor"): "\\textsc{syrk} $D_c = C_c + A_c A_c^\\top$",
        # ("RowMajor", "RowMajor"): "\\textsc{syrk} $D_r = C_r + A_r A_r^\\top$",
        # ("ColMajor", "RowMajor"): "\\textsc{syrk} $D_r = C_r + A_c A_c^\\top$",
    },
    "syrk_potrf": {
        ("RowMajor", "ColMajor"): "\\textsc{syrk+potrf} $D_c = \\mathrm{chol}(C_c + A_r A_r^\\top)$",
        ("ColMajor", "ColMajor"): "\\textsc{syrk+potrf} $D_c = \\mathrm{chol}(C_c + A_c A_c^\\top)$",
        ("RowMajor", "RowMajor"): "\\textsc{syrk+potrf} $D_r = \\mathrm{chol}(C_r + A_r A_r^\\top)$",
        ("ColMajor", "RowMajor"): "\\textsc{syrk+potrf} $D_r = \\mathrm{chol}(C_r + A_c A_c^\\top)$",
    },
    "trtri": {
        ("RowMajor",): "\\textsc{trtri} $D_r = L_r^{-1}$",
        ("ColMajor",): "\\textsc{trtri} $D_c = L_c^{-1}$",
    },
    "potrf": {
        ("RowMajor",): "\\textsc{potrf} $D_r = \\mathrm{chol}(A_r)$",
        ("ColMajor",): "\\textsc{potrf} $D_c = \\mathrm{chol}(A_c)$",
    },
    "hyh": {
        # ("RowMajor", "ColMajor"): "Hyhound $(\\tilde L_r \\; 0) = (L_r \\; A_c) \\breve Q$",
        ("ColMajor", "ColMajor"): "Hyhound $(\\tilde L_c \\; 0) = (L_c \\; A_c) \\breve Q$",
        ("RowMajor", "RowMajor"): "Hyhound $(\\tilde L_r \\; 0) = (L_r \\; A_r) \\breve Q$",
        # ("ColMajor", "RowMajor"): "Hyhound $(\\tilde L_c \\; 0) = (L_c \\; A_r) \\breve Q$",
    },
}

def make_subplots_grid(n):
    """Return sensible (rows, cols) for n subplots."""
    if n <= 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    elif n <= 4:
        return 2, 2
    elif n <= 6:
        return 2, 3
    elif n <= 9:
        return 3, 3
    else:
        rows = int(np.ceil(np.sqrt(n)))
        return rows, int(np.ceil(n / rows))

DO_FILL_BETWEEN = False

def plot_partitioned(df: pd.DataFrame, metric, stat, title, ylabel, relative=False, gflops=False, logx=False, x_lim_max=None):
    """Generic helper for all plots with subplots per partition."""
    for func_name, partitions in separate_figs.items():
        func_df = df[df["func"] == func_name]
        if func_df.empty:
            continue

        n_subplots = len(partitions)
        nrows, ncols = make_subplots_grid(n_subplots)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows + 0.6), squeeze=False, sharex="all", sharey="all")

        for ax, (args_tuple, subtitle) in zip(axes.flatten(), partitions.items()):
            # Select rows whose args start with the tuple args_tuple
            mask = func_df["args"].apply(lambda a: a[1:1 + len(args_tuple)] == args_tuple)
            sub_df = func_df[mask]
            if sub_df.empty:
                print("No data for", func_name, args_tuple)
                continue
            sub_df = sub_df.sort_values(by="label", kind="stable", key=lambda x: x.str.lower())

            for run in sub_df["run_name"].unique():
                run_df = sub_df[sub_df["run_name"] == run]
                opts = dict(label=run_df["label"].iloc[0], color=run_df["color"].iloc[0])

                if relative:
                    y_ref = run_df[f"{metric}_ref"][stat].array
                    y_val = run_df[metric][stat].array / y_ref
                    (pl,) = ax.loglog(run_df["version"], y_val, ".-", **opts)
                    if DO_FILL_BETWEEN:
                        ax.fill_between(
                            run_df["version"],
                            run_df[metric]["min"].array / y_ref,
                            run_df[metric]["max"].array / y_ref,
                            color=pl.get_color(),
                            alpha=0.25,
                        )
                elif gflops:
                    plotfun = ax.semilogx if logx else ax.plot
                    (pl,) = plotfun(run_df["version"], run_df["GFLOPS"][stat].array, ".-", **opts)
                    if DO_FILL_BETWEEN:
                        ax.fill_between(
                            run_df["version"],
                            run_df["GFLOPS"]["min"].array,
                            run_df["GFLOPS"]["max"].array,
                            color=pl.get_color(),
                            alpha=0.25,
                        )
                    ax.set_ylim(0, gflops_max)
                    if x_lim_max:
                        ax.set_xlim(1 if logx else 0, x_lim_max)
                else:
                    (pl,) = ax.loglog(run_df["version"], run_df[metric][stat], ".-", **opts)

            ax.set_title(subtitle)
            ax.legend(loc="lower right")
            ax.grid(True, which="both", ls=":")
        
        for r in range(nrows):
            axes[r, 0].set_ylabel(ylabel)

        for c in range(ncols):
            axes[-1, c].set_xlabel("Matrix size")

        # fig.suptitle(title, fontsize=15)
        plt.tight_layout()


def plot_absolute():
    plot_partitioned(
        df, metric, stat,
        f"Absolute Run Times of {title}",
        "Time (ns)",
        relative=False
    )


def plot_relative():
    plot_partitioned(
        df, metric, stat,
        f"Relative Run Times of {title}",
        "Time relative to MKL",
        relative=True
    )


def plot_gflops(log=False, x_lim_max=None):
    plot_partitioned(
        df, metric, stat,
        f"Performance of {title}",
        "Performance [GFLOPS]",
        gflops=True,
        logx=log,
        x_lim_max=x_lim_max
    )


plot_absolute()
plt.savefig(filename.with_suffix(".abs.pdf"))
plot_relative()
plt.savefig(filename.with_suffix(".rel.pdf"))
plot_gflops(True)
plt.savefig(filename.with_suffix(".gflops.pdf"))
plot_gflops(False, 64)
plt.savefig(filename.with_suffix(".gflops64.pdf"))
plot_gflops(False, 120)
plt.savefig(filename.with_suffix(".gflops120.pdf"))

plt.show()
