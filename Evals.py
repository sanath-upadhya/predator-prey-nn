"""
THIS FILE WRITTEN BY ADVAIT GOSAI
"""

from main import Model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import numpy as np
from Globals import *
import Loader

# From https://stackoverflow.com/questions/49880700/tensorflow-omp-error-15-when-training
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
    

parser = argparse.ArgumentParser(description='Process the .pkl filename.')
parser.add_argument('--filename', type=str, help='Name of the .pkl file to load')
args = parser.parse_args()

experiments = Loader.LoadPickled(args.filename)  # CHANGE THIS METHOD CALL APPROPRIATELY

end_reason_counts = {}
end_reason_plots = {}

unique_end_reasons = set()
markers = ['o', 's', '^', 'x']

for i, exp in enumerate(experiments):
    # Adding data to dictionary and plots for all experiments
    unique_end_reasons.add(exp['end_reason'])

    if exp['end_reason'] in end_reason_counts:
        end_reason_counts[exp['end_reason']] += 1
    else:
        end_reason_counts[exp['end_reason']] = 1
    if exp['end_reason'] not in end_reason_plots:
        end_reason_plots[exp['end_reason']] = []
    end_reason_plots[exp['end_reason']].append((i + 1, exp['sim_time']))

    # Intermediate plotting and printing for specific intervals
    if ((i % PRINT_EVAL_STEPS == 0) or (i == len(experiments) - 1)) and SHOW_LOSS_PLOTS:
        print(f"EXPERIMENT {i + 1}")
        try:
            preds = exp["PREDATORS"]
        except KeyError:
            preds = exp[PREDATOR]
        try:
            preys = exp["PREYS"]
        except KeyError:
            preys = exp[PREY]
        print(f"num_predators: {len(preds)}, num_preys: {len(preys)}")
        print(f"sim_time: {exp['sim_time']}")
        print(f"end_reason: {exp['end_reason']}")

        num_plots = 10 
        fig, axs = plt.subplots(num_plots // 2, 2, figsize=(15, 2 * num_plots))
        fig.suptitle(f"Predator Loss; Experiment {i+1}")  # ; {preds[0]['NETWORK'].name}
        for j in range(num_plots):
            row, col = divmod(j, 2)
            if j < len(preds):
                axs[row, col].plot(preds[j]["LOSSES"])
                axs[row, col].set_xlabel("Iteration")
                axs[row, col].set_ylabel("Losses")
        plt.tight_layout(pad=3.0)
        plt.show()

        fig, axs = plt.subplots(num_plots // 2, 2, figsize=(15, 2 * num_plots))
        fig.suptitle(f"Prey Loss; Experiment {i+1}")  # ; {preys[0]['NETWORK'].name}
        for j in range(num_plots):
            row, col = divmod(j, 2)
            if j < len(preys):
                axs[row, col].plot(preys[j]["LOSSES"])
                axs[row, col].set_title(f"Prey {j+1}")
                axs[row, col].set_xlabel("Iteration")
                axs[row, col].set_ylabel("Losses")
        plt.tight_layout(pad=3.0)
        plt.show()
        print("----------------------------")

print(end_reason_counts)

end_reason_marker = {reason: markers[i % len(markers)] for i, reason in enumerate(unique_end_reasons)}

plt.figure(figsize=(10, 6))
for reason, data in end_reason_plots.items():
    x, y = zip(*data)
    y = [round(float(y_i)/1000, 2) for y_i in y]
    plt.scatter(x, y, marker=end_reason_marker[reason], label=reason)

plt.title("End Reasons for Each Experiment")
plt.xlabel("Experiment Number")
plt.ylabel("Simulation Time")
plt.legend()
if SHOW_END_PLOTS:
    plt.show()
