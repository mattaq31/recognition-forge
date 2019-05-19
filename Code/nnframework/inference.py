from nnframework.model_architectures import FCCNetwork
import nnframework.data_builder as db
import constants as const
import os, re, torch
import numpy as np
from collections import OrderedDict

class Inference(object):
    """
        Initializes a fully connected network from a saved model and runs inference.
        :param model_file: The PyTorchf model file to load
        :param num_layers: The number of layers defined in model_file
        :param num_hidden_units: The number of hidden units defined in model_file
    """
    def __init__(self,  model_file, num_layers, num_hidden_units,):
        self.model_file = model_file
        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units

    """
        Run inference
    """
    def run(self, x):

        if x.ndim == 1:
            x = np.reshape(x, (1, len(x)))
            
        num_units = x.shape[0]
        features = x.shape[1]

        custom_fc_net = FCCNetwork(input_shape=(num_units, 1, 1, features),
            num_hidden_units=80,
            num_output_classes=3,
            use_bias=False,
            num_layers=5)

        # Load the custom object experiment builder saves
        state = torch.load(f=self.model_file)

        # Experiment builder is an NN module that wraps the FCCNetwork module in a property called "model"
        state_dict = OrderedDict()
        ex_state_dict = state['network']
        for key, val in ex_state_dict.items():
            new_key = re.sub("^model.", "", key)
            state_dict[new_key] = ex_state_dict[key]
            
        # Load the model
        custom_fc_net.load_state_dict(state_dict=state_dict)
        custom_fc_net.eval()

        x_tensor = torch.Tensor(x).float().to(device=torch.device('cpu'))
        
        result_tensor = custom_fc_net.forward(x_tensor)

        _, predictions = torch.max(result_tensor.data, 1)  # get argmax of predictions
        soft_fn = torch.nn.Softmax(dim=1)
        
        return predictions, result_tensor.detach().numpy(), soft_fn(result_tensor.detach()).numpy()