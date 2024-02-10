"""
THIS FILE WRITTEN BY RYAN FLETCHER, ADVAIT GOSAI, AND SANATH UPADHYA
"""

import torch
import math
from Globals import *
import Networks


class PredatorNetwork(Networks.CreatureVeryDeepFullyConnectedWithDropout):
    def __init__(self, hyperparameters, self_id):
        super().__init__(hyperparameters)
        self.base_lr = self.optimizer.param_groups[0]["lr"]
        self.min_lr = 0.1  # Minimum proportion of base_lr at any given step (see Networks.py line "new_lr = ...")
        self.id = self_id
    
    def transform(self, state_info):
        # Own energy + own other characteristics + other creatures' other characteristics IN THAT ORDER
        this = None
        for creature_state in state_info["creature_states"]:
            if creature_state["id"] == self.id:
                this = creature_state
                break
        flattened = [this[key] for key in self.self_inputs]  #[this["stun"], this["energy"], this["relative_speed"]]
        for prey_state in filter(FILTER_IN_PREY_DICTS, state_info["creature_states"]):
            flattened += [prey_state[key] for key in self.other_inputs]  #[prey_state["relative_speed"], prey_state["perceived_type"], prey_state["distance"]]
        return torch.FloatTensor(flattened)
    
    def loss(self, state_info):
        creature_states = filter(FILTER_IN_PERCEIVED_PREY_DICTS, state_info["creature_states"])
        if self.print_state:
            print(f"\nState info for {self.id}:\n\t{state_info['creature_states']}")
        closest = {"distance" : self.max_distance}  # Instantiated dynamically according to creature's sight range
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        r = torch.tensor(closest["distance"], requires_grad=True)
        if self.print_loss:
            print(f"\nLoss for {self.id}:\n\t{r}")
        return r, closest["distance"]
