from typing import Any

from torch import nn, Tensor

class CustomModule(nn.Module):
    def __init__(
        self,
        spec: dict[str, Any]
    ):
        '''
        returns pytorch.nn.Module according to set object

        Args:
            spec(dict[str, Any]): pytorch module spec. See below to look the rules: https://github.com/khhandrea/pytorch-initialize-module-from-yaml/README/md
        
        Returns
            module(pytorch.nn.Module): pytorch module
        '''
        super().__init__()
        module_list = []

        layer_idx = 0
        for layer_spec in spec['layers']:
            # Module
            if layer_spec['layer'] == 'linear':
                input_size, output_size = layer_spec['spec']
                module = nn.Linear(input_size, output_size)
            elif layer_spec['layer'] == 'conv2d':
                in_channels, out_channels, kernel_size, stride, padding = layer_spec['spec']
                module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            elif layer_spec['layer'] == 'rnn':
                input_size, hidden_size, num_layers = layer_spec['spec']
                module = nn.RNN(input_size, hidden_size, num_layers)
            elif layer_spec['layer'] == 'lstm':
                input_size, hidden_size, num_layers = layer_spec['spec']
                module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            elif layer_spec['layer'] == 'flatten':
                module = nn.Flatten()
            else:
                raise Exception(f"Invalid layer name: {layer_spec['layer']}")

            # Parameter initialization
            if layer_spec['layer'] in ('linear', 'conv2d'):
                if bool(spec['initialization']) == 'True':
                    if layer_spec['activation'] in ('relu', 'elu'):
                        nn.init.kaiming_uniform_(module.weight)
                    elif layer_spec['activation'] in ('softmax'):
                        nn.init.xavier_uniform_(module.weight)

            module_list.append(module)
            layer_idx += 1

            # Activation layer
            if layer_spec['layer'] in ('linear', 'conv2d'):
                if layer_spec['activation']:
                    if layer_spec['activation'] == 'relu':
                        activation = nn.ReLU()
                    elif layer_spec['activation'] == 'elu':
                        activation = nn.ELU()
                    elif layer_spec['activation'] == 'softmax':
                        activation = nn.Softmax(dim=1)
                    else:
                        raise Exception(f"Invalid activation: {layer_spec['activation']}")
                    module_list.append(activation)
                    layer_idx += 1

            self._layers = nn.ModuleList(module_list)

    def forward(self, x, state=None) -> Tensor:
        for layer in self._layers:
            if state is not None:
                x = layer(x, state)
                state = None
            else:
                x = layer(x)
        return x