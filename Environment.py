"""
THIS FILE WRITTEN BY RYAN FLETCHER
    ALLOW_PREDATOR_ENERGY_DEATH sections added and written by Sanath Upadhya
"""

import multiprocessing
from multiprocessing import Process
from queue import Empty
import time
import math
import numpy as np
from Globals import *
if DRAW:
    import pygame
import main


def rotate_vector(v, angle, DTYPE=DTYPE):
    rotation_matrix = np.array([
        [np.cos(angle, dtype=DTYPE), -np.sin(angle, dtype=DTYPE)],
        [np.sin(angle, dtype=DTYPE), np.cos(angle, dtype=DTYPE)]
    ], dtype=DTYPE)
    return rotation_matrix @ v


def copy_dir(from_vec, to_vec, a=None, DTYPE=DTYPE):
    if a is not None:
        angle = a
    else:
        b = NORMALIZE(from_vec)
        angle = ANGLE_BETWEEN(np.array(REFERENCE_ANGLE, dtype=DTYPE), b)
    return rotate_vector(to_vec, angle)


class Ray:
    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        
    def cast(self, screen, length, color=WHITE):
        # Currently does not account for space wrapping, though sight calculations do.
        # Update this purely-aesthetic method if we have time.
        end_point = self.position + ANGLE_TO_VEC(self.angle) * length
        pygame.draw.line(screen, color, self.position, end_point, 1)


class Creature:
    def __init__(self, params, model, creature_id=None):
        """
        :param params: See main
        :param model: main.Model
        :param creature_id: Optional int (will call main.get_id() by default)
        """
        self.id = creature_id if creature_id is not None else main.get_id()
        self.DTYPE = params["DTYPE"]
        attrs = model.attrs
        self.fov = attrs["fov"]
        self.sight_range = attrs["sight_range"]
        self.max_store_positions = params["store_positions"][0]
        self.store_positions_reference = params["store_positions"][1]
        self.positions = []
        self.directions = []
        self.alives = []
        self.mass = attrs["mass"]
        self.size = attrs["size"]
        self.max_forward_force = attrs["max_forward_force"]
        self.max_backward_force = attrs["max_backward_force"]
        self.max_sideways_force = attrs["max_lr_force"]
        self.max_rotation_force = attrs["max_rotate_force"]
        self.max_velocity = attrs["max_speed"]
        self.position = np.array([params["x"], params["y"]], dtype=self.DTYPE)
        self.velocity = np.array([0, 0], dtype=self.DTYPE)
        self.acceleration = np.array([0, 0], dtype=self.DTYPE)
        self.max_rotation_speed = attrs["max_rotate_speed"]
        self.initial_direction = params["initial_direction"]
        self.direction = self.initial_direction
        self.initial_energy = params["initial_energy"]
        self.energy = self.initial_energy
        self.force_energy_quotient = attrs["force_energy_quotient"]
        self.rotation_speed = 0.0
        self.old_rotation_speed = 0.0
        self.stun = 0.0
        self.rotation_energy_quotient = attrs["rotation_energy_quotient"]
        self.rays = [Ray(self.position, angle) for angle in np.linspace(-(self.fov / 2) * 2 * np.pi,
                                                                        (self.fov / 2) * 2 * np.pi,
                                                                        num=attrs["num_rays"], dtype=self.DTYPE)]
        self.model = model
        self.alive = True
        self.applied_force_this_tick = False
        self.motion_total = 0.0
        self.preys_eaten = 0
    
    def update_rotation_speed(self, speed):
        """
        :param speed: float ∝ radians per millisecond
        """
        self.rotation_speed = speed
        if abs(self.rotation_speed) > self.max_rotation_speed:
            self.rotation_speed = math.copysign(self.max_rotation_speed, self.rotation_speed)
    
    def rotate(self, delta_time):
        """
        :param delta_time: float milliseconds
        """
        energy_expenditure = abs(self.rotation_speed) * delta_time * self.rotation_energy_quotient
        # If creature tries to move when its energy is too low, it loses all its energy, can't move the way it wants, and is briefly stunned.
        stunned_this_tick = False
        if(energy_expenditure > 0.0) and (self.stun > 0.0) and (STUN_IGNORE_PUNISHMENT_QUOTIENT > 0.0):
            self.stun += STUN_TICK_TIME * STUN_IGNORE_PUNISHMENT_QUOTIENT
            energy_expenditure = self.energy
            stunned_this_tick = True
        elif (self.energy < energy_expenditure) and (self.stun == 0.0):
            self.stun += STUN_TICK_TIME
            energy_expenditure = self.energy
            stunned_this_tick = True
    
        if self.stun > 0.0:
            self.rotation_speed = 0.0
        
        recover_energy = False
        if ALLOW_PREDATOR_ENERGY_DEATH:
            if self.model.type == PREDATOR:
                pass
            else:
                recover_energy = True
        else:
            recover_energy = True

        if recover_energy:
            if (energy_expenditure == 0.0) or ((self.stun > 0.0) and not stunned_this_tick):
                self.stun = max(0, self.stun - delta_time)
                self.energy = min(self.initial_energy, self.energy + (abs(self.max_rotation_speed) * delta_time * self.rotation_energy_quotient))
            
        self.energy -= energy_expenditure
        for ray in self.rays:
            ray.angle = (ray.angle + (self.rotation_speed * delta_time)) % (2 * np.pi)
        self.direction = (self.direction + (self.rotation_speed * delta_time)) % (2 * np.pi)
        self.directions.append(self.direction)
    
    def apply_force(self, force, delta_time, self_motivated=True):
        """
        :param force: 2d-array : [ float forward force, float rightward force ]
        """
        f = force
        if self_motivated:
            f = np.clip(force, [-self.max_backward_force, -self.max_sideways_force],
                    [self.max_forward_force, self.max_sideways_force], dtype=self.DTYPE)
            energy_expenditure = np.linalg.norm(f) * self.force_energy_quotient
            # If creature tries to move when its energy is too low, it loses all its energy, can't move the way it wants, and is briefly stunned.
            stunned_this_tick = False
            if(energy_expenditure > 0.0) and (self.stun > 0.0) and (STUN_IGNORE_PUNISHMENT_QUOTIENT > 0.0):
                self.stun += STUN_TICK_TIME * STUN_IGNORE_PUNISHMENT_QUOTIENT
                energy_expenditure = self.energy
                stunned_this_tick = True
            elif (self.energy < energy_expenditure) and (self.stun == 0.0):
                self.stun += STUN_TICK_TIME
                energy_expenditure = self.energy
                stunned_this_tick = True
        
            if self.stun > 0.0:
                self.rotation_speed = 0.0
                f = np.array([0.0, 0.0])

            recover_energy = False
            if ALLOW_PREDATOR_ENERGY_DEATH:
                if self.model.type == PREDATOR:
                    pass
                else:
                    recover_energy = True
            else:
                recover_energy = True

            if recover_energy:
                if(energy_expenditure == 0.0) or ((self.stun > 0.0) and not stunned_this_tick):
                    self.stun = max(0, self.stun - delta_time)
                    self.energy = min(self.initial_energy, self.energy + (np.linalg.norm(np.array([max(self.max_forward_force, self.max_backward_force),
                                                                                                self.max_sideways_force]))
                                                                        * self.force_energy_quotient))
            
            self.energy -= energy_expenditure
        if np.linalg.norm(f) > 0.0:
            self.applied_force_this_tick = True
        self.acceleration += copy_dir(None, f / self.mass, a=self.direction)
    
    def update_velocity(self, delta_time):
        """
        :param delta_time: float milliseconds
        """
        velocity_magnitude = np.linalg.norm(self.velocity) / delta_time
        if (velocity_magnitude <= DRAG_MINIMUM_SPEED) and not self.applied_force_this_tick:
            self.velocity = np.array([0.0, 0.0])
        else:
            self.velocity += self.acceleration * delta_time
            if velocity_magnitude > self.max_velocity:
                self.velocity *= self.max_velocity / velocity_magnitude
    
    def update_position(self, env, delta_time):
        """
        :param delta_time: float milliseconds
        """
        old_position = np.array(self.position)
        self.position += self.velocity * delta_time
        self.acceleration = 0
        if self.position[0] < 0:
            self.position[0] = env.screen_width + self.position[0]
        if self.position[1] < 0:
            self.position[1] = env.screen_height + self.position[1]
        if env.screen_width - self.position[0] < 0:
            self.position[0] = -(env.screen_width - self.position[0])
        if env.screen_height - self.position[1] < 0:
            self.position[1] = -(env.screen_height - self.position[1])
        self.motion_total += np.linalg.norm(self.position - old_position)
        if self.max_store_positions > 0:
            if not ((self.store_positions_reference == FIRST) and (len(self.positions) >= self.max_store_positions)):
                self.positions.append(np.array(self.position))
        if (self.store_positions_reference == RECENT) and (len(self.positions) > self.max_store_positions):
            self.positions.pop(0)
    
    def draw(self, screen):
        for ray in self.rays:
            ray.cast(screen, self.sight_range, color=WHITE if self.alive else BACKGROUND_COLOR)
        pygame.draw.circle(screen, CREATURE_COLORS[self.model.type] if self.alive else GRAY, self.position.astype(int), self.size)
        self.alives.append(True if self.alive else False)
        bulge_radius = DEFAULT_CREATURE_SIZE
        bulge_position = self.position + ANGLE_TO_VEC(self.direction) * bulge_radius
        # Draw the bulge as a smaller circle or an arc
        bulge_size = 4
        pygame.draw.circle(screen, CREATURE_COLORS[self.model.type] if self.alive else GRAY, bulge_position.astype(int), bulge_size)
    
    def see_others(self, env):
        positions = [self.position]
        if self.position[0] - self.sight_range < 0:
            positions.append([env.screen_width + self.position[0], self.position[1]])
        if self.position[1] - self.sight_range < 0:
            positions.append([self.position[0], env.screen_height + self.position[1]])
        if env.screen_width - self.position[0] < self.sight_range:
            positions.append([-(env.screen_width - self.position[0]), self.position[1]])
        if env.screen_height - self.position[1] < self.sight_range:
            positions.append([self.position[0], -(env.screen_height - self.position[1])])
        can_see = []
        for creature in env.creatures:
            in_any_angle_and_range = False
            shortest_distance = math.inf
            for position in positions:
                distance = np.linalg.norm(creature.position - position)
                if (abs(ANGLE_BETWEEN(creature.position - position, ANGLE_TO_VEC(self.direction))) <= self.fov / 2) and\
                    (distance <= self.sight_range):
                    in_any_angle_and_range = True
                    if distance < shortest_distance:
                        shortest_distance = distance
            if (not (creature.id == self.id)) and in_any_angle_and_range and creature.alive:
                can_see.append((creature.id, shortest_distance))
        return can_see

    def get_results(self):
        self.model.epoch_losses.append(self.model.current_losses)
        losses = self.model.current_losses
        self.model.current_losses = []
        results = { "NETWORK" : self.model.NN, "LOSSES" : losses }
        if DEFAULT_STORE_CREATURE_POSITIONS > 0:
            results["POSITIONS"] = self.positions
            results["DIRECTIONS"] = self.directions
            results["ALIVES"] = self.alives
        if self.model.type == PREDATOR:
            results["PREYS_EATEN"] = self.preys_eaten
        # Add more analytics
        return results
    

def worker(task_queue, inputs_queue, creatures):
    while True:
        try:
            task = task_queue.get(block=False)
            if task is None:
                break
            else:
                creatures[task].model.get_inputs(task[0], queue=inputs_queue, index=task[1])
        except Empty:
            pass


class Environment:
    def __init__(self, env_params, models):
        self.DRAG_COEFFICIENT = env_params["DRAG_COEFFICIENT"]
        self.MIN_TPS = env_params["MIN_TPS"]
        self.EAT_EPSILON = env_params["EAT_EPSILON"]
        self.DTYPE = env_params["DTYPE"]
        self.num_preys = env_params["num_preys"]
        self.num_predators = env_params["num_predators"]
        self.screen_width = env_params["screen_width"]
        self.screen_height = env_params["screen_height"]
        self.creatures = []
        self.steps = 0
        self.time = 0.0
        for model in models:
            #print("FROM ENV 253: " + str(model[0]["attrs"]["sight_range"]))
            self.creatures.append(Creature(model[0], model[1], creature_id=model[1].NN.id))
        for creature in self.creatures:
            creature.model.creature = creature
            creature.model.environment = self
            creature.model.NN.max_distance = (float)(creature.sight_range)  # Convert to float for tensor typing compatibility
            creature.model.NN.max_velocity = (float)(creature.max_velocity)
        if USE_MULTIPROCESSING:
            self.task_queue = multiprocessing.Queue()
            self.inputs_queue = multiprocessing.Queue()
            self.expected_num_children = 0
            print("...")
            for _ in range(NUM_SUBPROCESSES):
                self.expected_num_children += 1
                process = Process(target=worker, args=(self.task_queue, self.inputs_queue, self.creatures))
                process.start()
            time.sleep(1)
            print(f"Expect {self.expected_num_children} \"True\"s")
            for process in multiprocessing.active_children():
                print(process.is_alive())
        print("... Beginning simulation")
        time.sleep(1)
        self.start_real_time = time.time()  # Currently overridden in Main.main()
    
    def step(self, delta_time, screen=None):        
        # Predation
        all_creature_pairs = [(a, b) for idx, a in enumerate(self.creatures) for b in self.creatures[idx + 1:]]  # From GeeksForGeeks
        for a, b in all_creature_pairs:
            if a.alive and b.alive and\
               (((a.model.type == PREY) and (b.model.type == PREDATOR)) or ((b.model.type == PREY) and (a.model.type == PREDATOR))) and\
               (np.linalg.norm(a.position - b.position) < ((1 - self.EAT_EPSILON) * (a.size + b.size))):
                if a.model.type == PREDATOR:
                    b.alive = False
                    a.energy += PREDATION_ENERGY_BOOST
                    a.preys_eaten += 1
                else:
                    #B is the predator here
                    a.alive = False
                    b.energy += PREDATION_ENERGY_BOOST
                    b.preys_eaten += 1

        all_prey_eaten = True
        for creature in filter(FILTER_IN_PREY_OBJECTS, self.creatures):
            if creature.alive:
                all_prey_eaten = False
                break
        if all_prey_eaten:
            print(ALL_PREYS_DEAD)
            return ALL_PREYS_DEAD

        if ALLOW_PREDATOR_ENERGY_DEATH:
            #Do the exact same thing as above, but for predators
            all_predators_dead = True
            for creature in filter(FILTER_IN_PREDATOR_OBJECTS, self.creatures):
                if creature.alive:
                    all_predators_dead = False
                    break

            if all_predators_dead:
                print(ALL_PREDATORS_DEAD)
                return ALL_PREDATORS_DEAD
        
        # Gather creature network feedback
        if USE_MULTIPROCESSING:
            all_inputs = [None]*len(self.creatures)
            for i in range(len(self.creatures)):
                self.task_queue.put((delta_time, i))
            while self.inputs_queue.qsize() < len(self.creatures):
                time.sleep(delta_time / 100)
            while not self.inputs_queue.empty():
                result = self.inputs_queue.get()
                all_inputs[result[0]] = result[1]
        else:
            all_inputs = [creature.model.get_inputs(delta_time) for creature in self.creatures]
        
        ########################################################################################
        # The following is for testing                                                         #
        ########################################################################################
        if DRAW:
            if self.creatures[FOCUS_CREATURE].alive:
                override = np.array([0.0, 0.0], dtype=self.DTYPE)
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    override += np.array([1.0, 0.0], dtype=self.DTYPE)
                if keys[pygame.K_a]:
                    override += np.array([0.0, -1.0], dtype=self.DTYPE)
                if keys[pygame.K_s]:
                    override += np.array([-1.0, 0.0], dtype=self.DTYPE)
                if keys[pygame.K_d]:
                    override += np.array([0.0, 1.0], dtype=self.DTYPE)
                if keys[pygame.K_SPACE]:
                    self.creatures[FOCUS_CREATURE].velocity = np.array([0.0, 0.0], dtype=self.DTYPE)
                override = NORMALIZE(override)
                if np.linalg.norm(override) > 0 or ALWAYS_OVERRIDE_PREY_MOVEMENT:
                    all_inputs[FOCUS_CREATURE][0] = override.tolist()
                override = 0.0
                if keys[pygame.K_LEFT]:
                    override += 2 * np.pi * (1 / 2) / 1000
                if keys[pygame.K_RIGHT]:
                    override += -2 * np.pi * (1 / 2) / 1000
                if keys[pygame.K_DOWN]:
                    override = -all_inputs[0][1]
                if ALWAYS_OVERRIDE_PREY_MOVEMENT:
                    all_inputs[FOCUS_CREATURE][1] = 0
                all_inputs[FOCUS_CREATURE][1] += override
        ########################################################################################
        # END TESTING                                                                          #
        ########################################################################################
        
        # Update creature positions
        for creature, inputs in zip(self.creatures, all_inputs):
            creature.applied_force_this_tick = False
            if creature.alive:
                creature.update_rotation_speed(-inputs[1])  # Negated because the screen is flipped
                creature.rotate(delta_time)
                if (np.linalg.norm(creature.velocity) / delta_time) > DRAG_MINIMUM_SPEED:
                    creature.apply_force(copy_dir(None,
                                                  -self.DRAG_COEFFICIENT * (np.linalg.norm(creature.velocity) ** 2) * NORMALIZE(creature.velocity),
                                                  a=-creature.direction),
                                         delta_time, self_motivated=False)
                creature.apply_force(np.array(inputs[0], dtype=self.DTYPE), delta_time)
                creature.update_velocity(delta_time)
                creature.update_position(self, delta_time)
            if DRAW:
                creature.draw(screen)
            
            if ALLOW_PREDATOR_ENERGY_DEATH:
                if (creature.energy < 0.0) and (creature.model.type == PREDATOR):
                    creature.alive = False
        
        # Timekeeping
        self.steps += 1
        self.time += delta_time
        if self.steps % PRINT_PROGRESS_STEPS == 0:
            current_real_time = time.time()
            print(f"{'' if self.experiment_set_name is None else (self.experiment_set_name + ': ')}Step # {self.steps}:\n\tSimulation time: {(self.time / 1000):.3f}s\n\tReal time: {(current_real_time - self.start_real_time):.3f}s\n\tTotal motions: {[creature.motion_total for creature in self.creatures]}")
        return SUCCESSFUL_STEP
    
    def get_state_info(self):
        """
        :return: { "creature_states" : [ see below ], "time" : float elapsed time }
        """
        state_info = {}
        creature_states = []
        for creature in self.creatures:
            creature_states.append({
                "type"              : creature.model.type,
                "position"          : creature.position,
                "direction"         : creature.direction,
                "velocity"          : creature.velocity,
                "id"                : creature.id,
                "energy"            : creature.energy,
                "initial_energy"    : creature.initial_energy,
                "stun"              : creature.stun
            })
        state_info["creature_states"] = creature_states
        state_info["time"] = self.time
        # TODO : WHAT OTHER STUFF?
        return state_info
