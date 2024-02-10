"""
THIS FILE WRITTEN BY ADVAIT GOSAI, RYAN FLETCHER, AND SANATH UPADHYA
"""

import torch
from Globals import *

dtype = DTYPE # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CreatureNetwork:
    def __init__(self, hyperparameters):
        self.model = None
        self.optimizer = None
        self.hyperparameters = hyperparameters
        self.self_inputs = hyperparameters["input_keys"][0]
        self.other_inputs = hyperparameters["input_keys"][1]
        self.print_state = hyperparameters["print_state"]
        self.print_loss = hyperparameters["print_loss"]

    def get_inputs(self, state_info):
        """
        :param state_info: See Environment.get_state_info() return specification.
        :return: [ 2d-array : [float forward force, float rightwards force], float clockwise rotation in [0,2pi) ], loss
        """
        # Transform into a 1d array with environment info first then info about all other relevent creatures.
        input = self.transform(state_info)
        scores, loss = self.train(self.model, self.optimizer, state_info, input)           
        return [[scores[0].item(), scores[1].item()], scores[2].item()], loss
    
    def train(self, model, optimizer, state_info, input):
        """
        Train a model on CIFAR-10 using the PyTorch Module API.
        
        Inputs:
        - model: A PyTorch Module giving the model to train.
        - optimizer: An Optimizer object we will use to train the model
        
        Returns: Nothing, but prints model accuracies during training.
        CURRENTLY STOLEN FROM HW2
        """
        model.train()  # put model to training mode
        model = model.to(device=device)  # move the model parameters to CPU/GPU
        
        scores = model(input)
        loss = self.loss(state_info)
        closest_dist = None
        try:
            loss, closest_dist = loss
        except TypeError:
            pass
        
        # Adjust learning rate dynamically so that creatures learn less during the long periods of time when they spend
        if ADJUST_LEARNING_RATE_WITH_DISTANCE and (closest_dist is not None):
            # Very high (caps at 2x around <=~33% of max distance) when fairly close, drops fast when distance approaches max
            new_lr = self.base_lr * min(2, max(self.min_lr, np.log((-2 * (closest_dist / self.max_distance)) + 2.25) + 1.5))
            for g in self.optimizer.param_groups:
                g['lr'] = new_lr

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()
        
        return scores, loss.item()

class CreatureFullyConnectedShallow(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        self.name = "Shallow Fully Connected with dimensions " + str(dims)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[1], dims[2])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        self.name = "Fully Connected with dimensions " + str(dims)        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[3], dims[4])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureDeepWithDropOut(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        self.name = "Deep FCN Dropout with dimensions " + str(dims) 
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[6], dims[7])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

# class DeepMLPWithLayerNorm(CreatureNetwork):
#     def __init__(self, hyperparameters):
#         super().__init__(hyperparameters)
#         dims = hyperparameters["dimensions"]
#         self.name = "Deep FCN LayerNorm with dimensions " + str(dims) 
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(dims[0], dims[1]),
#             torch.nn.LayerNorm(dims[1]),
#             torch.nn.LeakyReLU(0.01),
#             torch.nn.Linear(dims[1], dims[2]),
#             torch.nn.LayerNorm(dims[2]),
#             torch.nn.LeakyReLU(0.01),
#             torch.nn.Linear(dims[2], dims[3]),
#             torch.nn.LayerNorm(dims[3]),
#             torch.nn.LeakyReLU(0.01),
#             torch.nn.Linear(dims[3], dims[4]),
#             torch.nn.LayerNorm(dims[4]),
#             torch.nn.LeakyReLU(0.01),
#             torch.nn.Linear(dims[4], dims[5]),
#             torch.nn.LayerNorm(dims[5]),
#             torch.nn.LeakyReLU(0.01),
#             torch.nn.Linear(dims[5], dims[6]),
#             torch.nn.LayerNorm(dims[6]),
#             torch.nn.LeakyReLU(0.01),
#             torch.nn.Linear(dims[6], dims[7])
#         )
#         self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureVeryDeepFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        self.name = "Very Deep Fully Connected with dimensions " + str(dims) 

        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[6], dims[7]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[7], dims[8]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[8], dims[9]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[9], dims[10]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[10], dims[11]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[11], dims[12]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[12], dims[13]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[13], dims[14]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[14], dims[15]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[15], dims[16]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[16], dims[17]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[17], dims[18]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[18], dims[19]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[19], dims[20]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[20], dims[21]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[21], dims[22]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[22], dims[23]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[23], dims[24]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[24], dims[25]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[25], dims[26]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[26], dims[27]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[27], dims[28]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[28], dims[29]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[29], dims[30]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(dims[30], dims[31])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureVeryDeepFullyConnectedWithDropout(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        self.name = "Very Deep Fully Connected DropOut with dimensions " + str(dims) 

        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[6], dims[7]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[7], dims[8]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[8], dims[9]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[9], dims[10]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[10], dims[11]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[11], dims[12]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[12], dims[13]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[13], dims[14]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[14], dims[15]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[15], dims[16]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[16], dims[17]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[17], dims[18]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[18], dims[19]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[19], dims[20]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[20], dims[21]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[21], dims[22]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[22], dims[23]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[23], dims[24]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[24], dims[25]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[25], dims[26]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[26], dims[27]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[27], dims[28]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[28], dims[29]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[29], dims[30]),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[30], dims[31])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())



