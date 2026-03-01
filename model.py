import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm as weight_normalization # since weight_norm is a bool in our code

class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Leaky ReLU layers, Dropout layers and a tanh layer.
        # self.fc = nn.Linear(3, 1)
        # self.dropout_prob = dropout_prob
        # self.th = nn.Tanh()

        # We preffered this structure compared to the nn.Sequential because it is more customizable.

        self.fc1 = weight_normalization(nn.Linear(3, 512))
        self.fc2 = weight_normalization(nn.Linear(512, 512))
        self.fc3 = weight_normalization(nn.Linear(512, 512))
        self.fc4 = weight_normalization(nn.Linear(512, 509))

        self.fc5 = weight_normalization(nn.Linear(509 + 3, 512)) # We have to concatenate the input here during the forward pass.
        self.fc6 = weight_normalization(nn.Linear(512, 512))
        self.fc7 = weight_normalization(nn.Linear(512, 512))
        self.fc8 = nn.Linear(512, 1)

        self.activation = nn.LeakyReLU(0.01) # negative_slope picked from: https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        self.dropout = nn.Dropout(p=dropout_prob)
        self.th = nn.Tanh()

        # ***********************************************************************
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        # x = self.fc(input)
        # x = self.th(x)

        #input
        # The first 4 FC follow exactly the same logic (Except the last one that has an output of 509 instead of 512).
        x = self.dropout(self.activation(self.fc1(input)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        x = self.dropout(self.activation(self.fc4(x)))

        # Concatenate the output of the last FC with the input.
        # https://docs.pytorch.org/docs/stable/generated/torch.cat.html
        intermediate_tensor = torch.cat((x,input), dim=1)

        # The 3 FC below follow exactly the same logic.
        x = self.dropout(self.activation(self.fc5(intermediate_tensor)))
        x = self.dropout(self.activation(self.fc6(x)))
        x = self.dropout(self.activation(self.fc7(x)))

        # output is not passed through an activation function. Also, dropout is not added.
        x = self.th(self.fc8(x))

        # ***********************************************************************
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################

        return x
