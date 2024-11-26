import contextlib
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

filename = (
    Path(__file__).parent.parent
    / "results"
    / "eda2862a29b5b1dd477f905a676f132ba0a7322a"
    / "potrf-avx512.json"
)
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
gflops_max = 20

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

# Plot the data
df = df[np.logical_not(df["run_name"].str.startswith("dpotrf_base"))]
functions = df["run_name"].unique()
ref = df[df["run_name"].str.endswith("<scalar, MKLAll>")]

for function in functions:
    function_df = df[df["run_name"] == function]

    plt.loglog(
        function_df["version"],
        function_df[metric][stat],
        ".-",
        label=function,
    )

plt.title(f"Benchmark results {filename} (absolute)")
plt.xlabel("Size")
plt.ylabel("Time (ns)")
plt.legend()
plt.tight_layout()
plt.savefig(filename.with_suffix(".abs.pdf"))

plt.figure()

for function in functions:
    function_df = df[df["run_name"] == function]

    y_ref = ref[metric][stat].array
    (pl,) = plt.loglog(
        function_df["version"],
        function_df[metric][stat].array / y_ref,
        ".-",
        label=function,
    )
    plt.fill_between(
        function_df["version"],
        function_df[metric]["min"].array / y_ref,
        function_df[metric]["max"].array / y_ref,
        color=pl.get_color(),
        alpha=0.25,
    )

plt.title(f"Benchmark results {filename} (relative)")
plt.xlabel("Size")
plt.ylabel("Time relative to <scalar, MKLAll>")
plt.legend()
plt.tight_layout()
plt.savefig(filename.with_suffix(".rel.pdf"))


def plot_gflops(log=False, x_lim_max=None):
    plt.figure()
    for function in functions:
        function_df = df[df["run_name"] == function]

        metric = "GFLOPS"
        plotfun = plt.semilogx if log else plt.plot
        (pl,) = plotfun(
            function_df["version"],
            function_df[metric][stat].array,
            ".-",
            label=function,
        )
        plt.fill_between(
            function_df["version"],
            function_df[metric]["min"].array,
            function_df[metric]["max"].array,
            color=pl.get_color(),
            alpha=0.25,
        )
    plt.ylim(0, gflops_max)
    plt.xlim(1 if log else 0, x_lim_max)
    plt.title(f"Benchmark results {filename} (performance)")
    plt.xlabel("Size")
    plt.ylabel("Useful GFLOPS")
    plt.legend()
    plt.tight_layout()


plot_gflops(True)
plt.savefig(filename.with_suffix(".gflops.pdf"))
plot_gflops(False, 64)
plt.savefig(filename.with_suffix(".gflops64.pdf"))

plt.show()
