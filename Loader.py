from Globals import *
import pickle


def LoadPickled(filename):
    return pickle.load(open(filename, "rb"))


def LoadPlaintext(filename):
    data = open(filename, "r")
    experiment_strs = []
    # Num prey, num predators, sim time, end reason, losses per creature, positions per creature, number of preys eaten per predator later
    experiments = []
    data_str = data.read().replace(' ', '')
    end = 0
    # CORRECT vvv
    while(end > -1):
        start = data_str.find("{\'real_time", end + 1)
        end = max(data_str.find("{\'real_time", start + 1) - 1, -1)
        experiment_strs.append(data_str[start:end])

    for exp_str in experiment_strs:
        last_index = 0
        index = -1
        experiment = {}
        index = exp_str.find("sim_time\':")
        comma_index = exp_str.find(",", index + len("sim_time\':"))
        experiment["sim_time"] = exp_str[index + len("sim_time\':") : comma_index]
        index = exp_str.find("end_reason\':\'")
        apos_index = exp_str.find("\'", index + len("end_reason\':\'"))
        experiment["end_reason"] = exp_str[index + len("end_reason\':\'") : apos_index]
        preys = []
        predators = exp_str.find("," + str(PREDATOR) + ":")
        last_index = 0
        index = -1
        while(index < predators):
            prey = {}
            index = exp_str.find("{", last_index + 1)
            loss_index = exp_str.find("LOSSES\':", index) + len("LOSSES\':")
            positions_index = exp_str.find("POSITIONS\':", loss_index)
            loss_end = positions_index - 2
            directions_index = exp_str.find("DIRECTIONS\':", positions_index) + len("DIRECTIONS\':")
            positions_end = directions_index - len("DIRECTIONS\':") - 2
            prey["LOSSES"] = list(map(lambda x : float(x), exp_str[loss_index:loss_end][1:-1].split(',')))
            positions_index += len("POSITIONS\':")
            units = exp_str[positions_index:positions_end][1:-1].split(',array')
            units[0] = units[0][5:]
            prey["POSITIONS"] = list(map(lambda x : np.array(list(map(lambda y : float(y), x[1:-1][1:-1].split(',')))), units))
            directions_end = exp_str.find("]", directions_index)
            prey["DIRECTIONS"] = list(map(lambda x : float(x), exp_str[directions_index:directions_end][1:-1].split(",")))
            preys.append(prey)
            last_index = directions_end
            
        index = predators
        predators = []
        last_index = index
        while(last_index < len(exp_str) - 6):
            predator = {}
            index = exp_str.find("{", last_index + 1)
            if index < 0:
                break
            loss_index = exp_str.find("LOSSES\':", index) + len("LOSSES\':")
            positions_index = exp_str.find("POSITIONS\':", loss_index)
            loss_end = positions_index - 2
            directions_index = exp_str.find("DIRECTIONS\':", positions_index) + len("DIRECTIONS\':")
            positions_end = directions_index - len("DIRECTIONS\':") - 2
            predator["LOSSES"] = list(map(lambda x : float(x), exp_str[loss_index:loss_end][1:-1].split(',')))
            positions_index +=  len("POSITIONS\':")
            units = exp_str[positions_index:positions_end][1:-1].split(',array')
            units[0] = units[0][5:]
            predator["POSITIONS"] = list(map(lambda x : np.array(list(map(lambda y : float(y), x[1:-1][1:-1].split(',')))), units))
            directions_end = exp_str.find("]", directions_index)
            predator["DIRECTIONS"] = list(map(lambda x : float(x), exp_str[directions_index:directions_end][1:-1].split(",")))
            predators.append(predator)
            last_index = directions_end
        experiment["PREYS"] = preys
        experiment["PREDATORS"] = predators
        experiment["num_preys"] = len(preys)
        experiment["num_predators"] = len(predators)
        experiments.append(experiment)
    
    with open("test.txt", "w") as fileeee:
        fileeee.write(str(experiments[0]["PREYS"][0]["POSITIONS"]))
    return experiments