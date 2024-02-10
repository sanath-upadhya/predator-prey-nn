"""
THIS FILE WRITTEN BY RYAN FLETCHER
"""

import argparse
import multiprocessing
import pickle
import numpy as np
import math
import random
import copy
import time
from Globals import *
if DRAW:
    import pygame
import Environment
import PreyNetwork
import PredatorNetwork
import Networks

id_count = 0
SPEED_ESTIMATION_DECAY = 1.0  # float in [0,inf). 0 means perfect speed estimation,
                              #                   higher k means worse speed estimation past sight_distance


def get_id():
    global id_count
    id_count += 1
    return id_count


class Model:
    def __init__(self, creature_type, attrs, hyperparameters, network=None):
        self.type = creature_type
        new_id = get_id()
        if not (network is None):
            self.NN = network
        elif creature_type == PREY:
            self.NN = PreyNetwork.PreyNetwork(hyperparameters, new_id)
        elif creature_type == PREDATOR:
            self.NN = PredatorNetwork.PredatorNetwork(hyperparameters, new_id)
        else:
            self.NN = None
        if self.NN is not None:
            self.NN.model_object = self
        self.sight_range = attrs["sight_range"]
        self.mass = attrs["mass"]
        self.size = attrs["size"]
        self.max_forward_force = attrs["max_forward_force"]
        self.max_backward_force = attrs["max_backward_force"]
        self.max_sideways_force = attrs["max_lr_force"]
        self.max_rotation_force = attrs["max_rotate_force"]
        self.max_velocity = attrs["max_speed"]
        self.max_rotation_speed = attrs["max_rotate_speed"]
        self.fov = attrs["fov"]
        self.creature = None  # Instantiate dynamically
        self.environment = None  # Instantiate dynamically
        self.epoch_losses = []
        self.current_losses = []
        self.attrs = attrs
    
    def get_inputs(self, delta_time, queue=None, index=None):
        state_info = self.environment.get_state_info()
        relative_state_info = {"time": state_info["time"]}
        creature_states = state_info["creature_states"]
        sights = self.creature.see_others(self.environment)  # [(id, distance), (id, distance), ...]
        relative_creature_states = []
        for state in creature_states:
            hit = False
            if not (state["id"] == self.creature.id):
                for sight in sights:
                    if sight[0] == state["id"]:
                        hit = True
                        perceived_type = state["type"]
                        distance = sight[1]
                        decay = min(1, max(0, math.exp(-SPEED_ESTIMATION_DECAY * ((distance / (self.creature.sight_range / 2)) - 1))))
                        relative_speed_x = (state["velocity"][0] - self.creature.velocity[0]) * decay / (self.creature.max_velocity * 2 * delta_time)
                        relative_speed_y = (state["velocity"][1] - self.creature.velocity[1]) * decay / (self.creature.max_velocity * 2 * delta_time)
            if not hit:
                # These may need to be implemented differently if the networks don't like implicit multimodality
                perceived_type = UNKNOWN_TYPE
                distance = self.creature.sight_range
                relative_speed_x = 0.0
                relative_speed_y = 0.0
                
            relative_creature_states.append({
                "type"              : state["type"],
                "perceived_type"    : perceived_type,
                "distance"          : distance / self.creature.sight_range,
                "position"          : state["position"],
                "relative_speed_x"  : relative_speed_x,
                "relative_speed_y"  : relative_speed_y,
                "id"                : state["id"],
                "energy"            : state["energy"] / state["initial_energy"],
                "stun"              : state["stun"],
                "relative_angle"    : ANGLE_BETWEEN(self.creature.direction, state["direction"])
            })
        relative_state_info["creature_states"] = relative_creature_states
        inputs, loss = self.NN.get_inputs(relative_state_info) if self.creature.alive else NETWORK_OUTPUT_DEFAULT
        self.current_losses.append(loss)
        if (queue is None) or (index is None):
            return inputs
        queue.put((index, inputs))


def main(serialize=True, name=None, new_allow_energy_death=ALLOW_PREDATOR_ENERGY_DEATH, prey_loss_mode=DEFAULT_PREY_LOSS_MODE, default_prey_network=None, default_predator_network=None):
    """
        Instructions for experimentation:
        Set PreyNetwork.PreyNetwork parent class to the appropriate Network.
        Set PredatorNetwork.PredatorNetwork parent class to the appropriate Network.
        Set Globals.PREY_NETWORK_HYPERMARATERS["dimensions"] = <appropriate dimensions list>
        Set Globals.PREDATOR_NETWORK_HYPERMARATERS["dimensions"] = <appropriate dimensions list>
        Comment/uncomment the two flags approrpriately at the top of Main.main().
        Run in the terminal:
            python3 main.py --name "<a descriptive name for the serialization file>"
    """
    experiments = []
    previous_experiment = DEFAULT_EXPERIMENT
    max_max_sim_time = previous_experiment[MAX_SIM_SECONDS]
    ALLOW_PREDATOR_ENERGY_DEATH = new_allow_energy_death  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< THIS IS ONE EXPERIMENT MODE FLAG!
    for i in range(100):
        #############################################################################################################
        experiment = copy.deepcopy(previous_experiment)
        #############################################################################################################
        
        # COMMENT THE FOLLOWING LINE WHEN NOT TESTING
        # experiment[MAX_SIM_SECONDS] = 30
        
        experiment[PREY_HYPERPARAMS_NAME]["loss_mode"] = prey_loss_mode  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< THIS IS ONE EXPERIMENT MODE FLAG!
        
        # Modify experiment parameters
        # In these experiments we're starting the creatures off with low energy so they can learn what it means.
        # We're also starting with a smaller screen size so loss values are bigger early on to encourage learning control.
        experiment[PREY_PARAMS_NAME]["initial_energy"] = min(5, math.pow(1.03, i))  # Should hit 100 at the 55th experiment
        experiment[PREDATOR_PARAMS_NAME]["initial_energy"] = min(5, math.pow(1.03, i))  # Should hit 100 at the 55th experiment
        experiment[PREY_ATTRS_NAME]["sight_range"] = int(0.75 * ((min(experiment[ENV_PARAMS_NAME]["screen_width"], experiment[ENV_PARAMS_NAME]["screen_height"]) * (2.0 / 3.0)) - max(experiment[PREY_ATTRS_NAME]["size"], experiment[PREDATOR_ATTRS_NAME]["size"])))
        experiment[PREDATOR_ATTRS_NAME]["sight_range"] = int((min(experiment[ENV_PARAMS_NAME]["screen_width"], experiment[ENV_PARAMS_NAME]["screen_height"]) * (2.0 / 3.0)) - max(experiment[PREY_ATTRS_NAME]["size"], experiment[PREDATOR_ATTRS_NAME]["size"]))
        #experiment[ENV_PARAMS_NAME]["screen_width"] = min(DEFAULT_SCREEN_WIDTH, 300 * ((i / 4.0) + 1))
        #experiment[ENV_PARAMS_NAME]["screen_height"] = min(DEFAULT_SCREEN_HEIGHT, 300 * ((i / 4.0) + 1))
        experiment[MAX_SIM_SECONDS] = min(max_max_sim_time, int((max_max_sim_time * 1.04) / (1 + math.exp(-(i - 50) / 15))))  # CHECK WHEN CHANGING DEFAULT MAX SIM TIME
        #############################################################################################################
        experiments.append(experiment)
        previous_experiment = experiment
        #############################################################################################################
    
    experiment_results = []
    for i in range(len(experiments)):
        print(f"Starting experiment {i + 1}")
        experiment = experiments[i]
        #print("Experiment details:\n\t" + "Max sim seconds: " + str(experiment[MAX_SIM_SECONDS]))
        
        if DRAW:
            pygame.init()
            screen = pygame.display.set_mode((experiment[ENV_PARAMS_NAME]["screen_width"], experiment[ENV_PARAMS_NAME]["screen_height"]))
            clock = pygame.time.Clock()
        
        running = True
        
        # Currently specified relative to the manually-set default x and y in <>__PARAMS in Globals
        num_preys = experiment[ENV_PARAMS_NAME]["num_preys"]
        num_predators = experiment[ENV_PARAMS_NAME]["num_predators"]
        models = [(copy.deepcopy(experiment[PREY_PARAMS_NAME]), Model(PREY, copy.deepcopy(experiment[PREY_ATTRS_NAME]), experiment[PREY_HYPERPARAMS_NAME], network=None if ((not experiment[KEEP_WEIGHTS]) or (i == 0)) else experiment_results[i - 1][PREY][j % len(experiment_results[i - 1][PREY])]["NETWORK"])) for j in range(num_preys)] +\
                 [(copy.deepcopy(experiment[PREDATOR_PARAMS_NAME]), Model(PREDATOR, copy.deepcopy(experiment[PREDATOR_ATTRS_NAME]), experiment[PREDATOR_HYPERPARAMS_NAME], network=None if ((not experiment[KEEP_WEIGHTS]) or (i == 0)) else experiment_results[i - 1][PREDATOR][j % len(experiment_results[i - 1][PREDATOR])]["NETWORK"])) for j in range(num_predators)]
        #print("ATTRS from model\n\n" + str(models[3][1].sight_range) + "\n\n")
        
        env = Environment.Environment(experiment[ENV_PARAMS_NAME], models)
        
        screen_width = env.screen_width
        screen_height = env.screen_height
        for i in range(num_preys + num_predators):
            tryX = None
            tryY = None
            too_close = True
            while too_close:
                tryX = 1 + (random.random() * (screen_width - 2))
                tryY = 1 + (random.random() * (screen_height - 2))
                too_close = False
                for model in models:
                    if np.linalg.norm(np.array([tryX, tryY], dtype=experiment[ENV_PARAMS_NAME]["DTYPE"]) - np.array([model[0]["x"], model[0]["y"]], dtype=experiment[ENV_PARAMS_NAME]["DTYPE"])) < DEFAULT_CREATURE_SIZE + PLACEMENT_BUFFER:
                        too_close = True
                        break
            models[i][0]["x"] = tryX
            models[i][0]["y"] = tryY
        
        if DRAW:
            focus_creature = FOCUS_CREATURE  # Index of focused creature in environment's list
            focus_pos = []
        
        env.start_real_time = time.time()

        end_reason = None
        while running:
            try:
                if DRAW:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            end_reason = "TERMINATED"
                    screen.fill(BACKGROUND_COLOR)
                    delta_time = min(1.0 / env.MIN_TPS * 1000, clock.tick(MAX_TPS))
                    # print(delta_time * MAX_TPS / 1000)  # ~=1 if on target TPS
                else:
                    
                    delta_time = 1.0 / MAX_TPS * 1000
                
                step_result = env.step(delta_time, screen=screen if DRAW else None)
                
                ##################################################################################
                # This is for testing.                                                           #
                ##################################################################################
                if DRAW:
                    if(env.creatures[focus_creature].alive):
                        focus_pos.append(env.creatures[focus_creature].position.tolist())
                    for i in range(min(len(focus_pos), FOCUS_PATH_LENGTH)):
                        pygame.draw.circle(screen, (np.array(GREEN, dtype=DTYPE) * i / min(len(focus_pos), FOCUS_PATH_LENGTH)).tolist(),
                                        (int(focus_pos[max(0, len(focus_pos) - FOCUS_PATH_LENGTH) + i][0]),
                                            int(focus_pos[max(0, len(focus_pos) - FOCUS_PATH_LENGTH) + i][1])),
                                        2)
                    env.creatures[focus_creature].draw(screen)
                ##################################################################################
                # End testing                                                                    #
                ##################################################################################
                
                if step_result == ALL_PREYS_DEAD:
                    running = False
                    end_reason = ALL_PREYS_DEAD

                if step_result == ALL_PREDATORS_DEAD:
                    running = False
                    end_reason = ALL_PREDATORS_DEAD

                if DRAW:
                    pygame.display.flip()
            except KeyboardInterrupt:
                running = False
                end_reason = "TERMINATED"
                print("Caught KeyboardInterrupt")
            if (env.time / 1000) >= experiment[MAX_SIM_SECONDS]:
                running = False
                end_reason = MAX_SIM_SECONDS
        
        results = {}
        results["real_time"] = time.time() - env.start_real_time
        results["sim_time"] = env.time
        results["end_reason"] = end_reason
        results[PREY] = [ creature.get_results() for creature in filter(FILTER_IN_PREY_OBJECTS, env.creatures) ]
        results[PREDATOR] = [ creature.get_results() for creature in filter(FILTER_IN_PREDATOR_OBJECTS, env.creatures) ]
        experiment_results.append(results)
        
        if DRAW:
            pygame.quit()
        if USE_MULTIPROCESSING:
            for _ in range(env.expected_num_children):
                    env.task_queue.put(None)
            time.sleep(.1)
            for process in multiprocessing.active_children():
                    process.join()
        
        if end_reason == "TERMINATED":
            break

    # COMMENT THIS LINE EXCEPT FOR TESTING
    # print(experiment_results)
    
    total_sim_time = 0.0
    total_real_time = 0.0
    for i in range(len(experiment_results)):
        total_sim_time += experiment_results[i]["sim_time"]
        total_real_time += experiment_results[i]["real_time"]
        print(f"End reason {i + 1}: {experiment_results[i]['end_reason']}")
    print(f"Total simulated time:    {int((total_sim_time / 1000) // 3600)}h {int(((total_sim_time / 1000) % 3600) // 60)}m {((((total_sim_time / 1000) % 3600) % 60)):.3f}s\nTotal real time:         {int(total_real_time // 3600)}h {int((total_real_time % 3600) // 60)}m {(((total_real_time % 3600) % 60)):.3f}s")
    
    if serialize:
        idn = "_" + str(int(time.time() - 1701300000))
        filename = ('data/' + (name + "_") if name is not None else "") + 'serialized_data' + idn + '.pkl'
        print("Serialized file name:\n\t" + filename)
        with open(filename, 'wb') as file:
            try:
                pickle.dump(experiment_results, file)
            except:
                print("\n\nPickling error. SERIALIZING MANUALLY! NETWORK WEIGHTS WILL NOT BE STORED!\n\n")
                with open(filename, 'w') as filee:
                    filee.write(str(experiment_results))
    # To read experiment_results later:
    # with open('serialized_data' + idn + '.pkl', 'rb') as file:
    #   loaded_object = pickle.load(file)


if __name__ == "__main__":
    print("Setting up...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Optional argument", default=None)
    parser.add_argument("--serialize", type=str, help="Optional argument", default="True")
    args = parser.parse_args()
    
    network_names = ["VERY_DEEP_DROPOUT_NETWORK_1_YESENERGY_RECIPROCAL_100_INITIAL_ENERGY_5_CHANGE_IN_MAIN"]
    network_hyper_hyper_params = [VERY_DEEP_DROPOUT_NETWORK_1_YESENERGY_RECIPROCAL]
    for name, network in zip(network_names, network_hyper_hyper_params):
        DEFAULT_PREY_NETWORK_HYPERPARAMETERS["dimensions"] = network[0]
        DEFAULT_PREDATOR_NETWORK_HYPERPARAMETERS["dimensions"] = network[1]
        main(name=(args.name + "_" if args.name is not None else "") + name, new_allow_energy_death=network[2], prey_loss_mode=network[3])
    # DEFAULT_PREY_NETWORK_HYPERPARAMETERS["dimensions"] = network_hyper_hyper_params[6][0]
    # DEFAULT_PREDATOR_NETWORK_HYPERPARAMETERS["dimensions"] = network_hyper_hyper_params[6][1]
    # main(name=(args.name + "_" if args.name is not None else "") + network_names[6], new_allow_energy_death=network_hyper_hyper_params[6][2], prey_loss_mode=network_hyper_hyper_params[6][3])
