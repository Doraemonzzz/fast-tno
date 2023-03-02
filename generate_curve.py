import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

sns.set()


def get_stat(file_path):
    stat = {}
    stat["CPU"] = {}
    stat["GPU"] = {}
    for stage in ["forward", "backward"]:
        stat["CPU"][stage] = {}
        stat["GPU"][stage] = {}

    with open(file_path) as f:
        for data in f.readlines():
            if " forward" in data or " backward" in data:
                name, stage = data.strip().split()
            if "CPU time total" in data or "CUDA time total" in data:
                if "CPU" in data:
                    hardware = "CPU"
                else:
                    hardware = "GPU"

                string = data.split()[-1]
                if "us" in string:
                    time = float(string[:-2])
                elif "ms" in string:
                    time = 100 * float(string[:-2])
                else:
                    time = 10000 * float(string[:-2])
                stat[hardware][stage][name] = time

    return stat


def get_stat_loop(prefix, dir="log"):
    data = {}
    for file in os.listdir(dir):
        if prefix in file:
            n = int(file.split("_")[-1].split(".")[0])
            file_path = os.path.join(dir, file)
            stat = get_stat(file_path)
            data[n] = stat

    return data


def draw(stat, prefix="n"):
    seq = sorted(list(stat.keys()))
    res = dict()
    for hardware in ["CPU", "GPU"]:
        res[hardware] = dict()
        for mode in ["forward", "backward"]:
            res[hardware][mode] = dict()

    for i in seq:
        for hardware in ["CPU", "GPU"]:
            for mode in ["forward", "backward"]:
                for name in stat[i][hardware][mode]:
                    if name not in res[hardware][mode]:
                        res[hardware][mode][name] = []
                    res[hardware][mode][name].append(stat[i][hardware][mode][name])

    for hardware in ["CPU", "GPU"]:
        for mode in ["forward", "backward"]:
            for name in res[hardware][mode]:
                sns.lineplot(
                    x=seq,
                    y=np.log(res[hardware][mode][name]),
                    label=name,
                    marker="o",
                    markersize=4,
                )

            plt.xlabel(rf"${prefix}$")
            plt.ylabel(r"log(us)")
            plt.title(f"{hardware} {mode}: {prefix} vs time")
            plt.legend()
            plt.savefig(f"image/{prefix}_{hardware}_{mode}.jpg", bbox_inches="tight")
            plt.close()


n_stat = get_stat_loop("n_test")
draw(n_stat, "n")

d_stat = get_stat_loop("d_test")
draw(d_stat, "d")

b_stat = get_stat_loop("b_test")
draw(b_stat, "b")