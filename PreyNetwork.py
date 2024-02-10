"""
THIS FILE WRITTEN BY RYAN FLETCHER, SANATH UPADHYA, AND ADVAIT GOSAI
"""

import torch
import math
from Globals import *
import Networks


class PreyNetwork(Networks.CreatureVeryDeepFullyConnectedWithDropout):
    def __init__(self, hyperparameters, self_id):
        super().__init__(hyperparameters)
        self.base_lr = self.optimizer.param_groups[0]["lr"]
        self.min_lr = 0.15  # Minimum proportion of base_lr at any given step (see Networks.py line "new_lr = ...")
        self.loss_mode = hyperparameters.get("loss_mode", SUBTRACT_MODE)
        self.id = self_id
        
    def transform(self, state_info):
        # Own energy + own other characteristics + other creatures' other characteristics IN THAT ORDER
        this = None
        for creature_state in state_info["creature_states"]:
            if creature_state["id"] == self.id:
                this = creature_state
                break
        flattened = [this[key] for key in self.self_inputs]  #[this["stun"], this["energy"], this["relative_speed"]]
        for predator_state in filter(FILTER_IN_PREDATOR_DICTS, state_info["creature_states"]):
            flattened += [predator_state[key] for key in self.other_inputs]  #[predator_state["relative_speed"], predator_state["perceived_type"], predator_state["distance"]]
        return torch.FloatTensor(flattened)
    
    def loss(self, state_info):
        creature_states = filter(FILTER_IN_PERCEIVED_PREDATOR_DICTS, state_info["creature_states"])
        self_position = np.array([-1.0, -1.0])
        for state in state_info["creature_states"]:
            if state["id"] == self.id:
                self_position = state["position"]
        if self.print_state:
            print(f"\nState info for {self.id}:\n\t{state_info['creature_states']}")
        closest = { "distance" : self.max_distance, "relative_speed_x" : 0, "relative_speed_y" : 0 }  # Instantiated dynamically according to creature's sight range
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        
        loss = None
        if self.loss_mode == RECIPROCAL_MODE:
            loss = torch.tensor(1.0 / closest["distance"], requires_grad=True)
            if self.print_loss:
                print(f"\nLoss for {self.id}:\n\t{loss}")
        elif self.loss_mode == SUBTRACT_MODE:
            loss = torch.tensor(-closest["distance"] + self.max_distance, requires_grad=True)
            if self.print_loss:
                print(f"\nLoss for {self.id}:\n\t{loss}")
        elif self.loss_mode == BIASED_SUBTRACT_MODE:  # Written by Advait Gosai, modified by Ryan Fletcher
            # DOES NOT WORK, DO NOT USE
            dif = closest["relative_speed_x"] + closest["relative_speed_y"]
            theta = ANGLE_BETWEEN(self_position, closest["position"])
            relative_speed = np.linalg.norm(np.sin((np.pi / 2) - theta) * dif)  # NEEDS TESTING, but pretty sure it's correct
            # sign = (closest["relative_speed_x"] + closest["relative_speed_y"]) / abs(closest["relative_speed_x"] + closest["relative_speed_y"])
            # relative_speed = sign * math.sqrt((closest["relative_speed_x"] ** 2) + (closest["relative_speed_y"] ** 2))
            bias = (self.max_distance / 20) * (1 - (relative_speed / self.max_velocity))
            loss = torch.tensor(-closest["distance"] + self.max_distance + bias, requires_grad=True)
            if self.print_loss:
                print(f"\nLoss for {self.id}:\n\t{loss}")
                
        return loss, closest["distance"]
