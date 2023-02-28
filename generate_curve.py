import matplotlib.pyplot as plt
import os

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
                else:
                    time = 100 * float(string[:-2])
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

def draw(stat):
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

    
                    

stat = get_stat_loop("n_test")
print(stat)
print(sorted(list(stat.keys())))
draw(stat)