import contextlib
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt

filename = "../gemm-omp1.json"
with contextlib.suppress(IndexError):
    filename = sys.argv[1]

# Load the JSON data
with open(filename) as f:
    data = json.load(f)

# Create a DataFrame from the benchmarks
df = pd.DataFrame(data["benchmarks"])

# Extract the final part of the name after the last slash for the x-axis
df["version"] = df["name"].apply(lambda x: int(x.split("/", 2)[1]))
df.sort_values(by="version", inplace=True)

# Plot the data
df["full name"] = df["name"]
df["name"] = df["name"].apply(lambda x: x.split("/", 2)[0])
functions = df["name"].unique()

for function in functions:
    function_df = df[df["name"] == function]

    for metric in ["real_time"]:
        plt.loglog(
            function_df["version"],
            function_df[metric],
            ".-",
            label=function,
        )

plt.title(f"Benchmark results {filename} (absolute)")
plt.xlabel("Size")
plt.ylabel("Time (ns)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure()

ref = df[df["name"].str.endswith("<scalar, Reference>")]
for function in functions:
    function_df = df[df["name"] == function]

    for metric in ["real_time"]:
        plt.loglog(
            function_df["version"],
            function_df[metric].array / ref[metric].array,
            ".-",
            label=function,
        )

plt.title(f"Benchmark results {filename} (relative)")
plt.xlabel("Size")
plt.ylabel("Time relative to <scalar, Reference>")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
