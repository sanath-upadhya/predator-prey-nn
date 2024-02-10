import os
import sys

path = os.getcwd() + "/" + "data/12 3-4 Experiments"
pkls = os.listdir(path)
evalsfile = open("Evals.py", "r")
evals = evalsfile.read()
for pkl in pkls:
    print(pkl)
    sys.argv = ["filename", path + "/" + pkl]
    exec(evals)
evalsfile.close()