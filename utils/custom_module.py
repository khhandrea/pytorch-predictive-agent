from torch import nn, Tensor

class CustomModule(nn.Module):
    def __init__(
        self,
        spec: dict[str, dict]
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

        for layer_spec in spec['layers']:
            # Module
            match layer_spec['layer']:
                case 'linear':
                    input_size, output_size = layer_spec['spec']
                    module = nn.Linear(input_size, output_size)
                case 'conv2d':
                    in_channels, out_channels, kernel_size, stride, padding = layer_spec['spec']
                    module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                case 'rnn':
                    input_size, hidden_size, num_layers = layer_spec['spec']
                    module = nn.RNN(input_size, hidden_size, num_layers)
                case 'lstm':
                    input_size, hidden_size, num_layers = layer_spec['spec']
                    module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                case 'gru':
                    input_size, hidden_size, num_layers = layer_spec['spec']
                    module = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
                case 'flatten':
                    module = nn.Flatten()
                case _:
                    raise Exception(f"Invalid layer name: {layer_spec['layer']}")

            # Parameter initialization
            kaiming_type_activation = ('relu', 'elu')
            xavier_type_activation = ('softmax',)
            if spec.get('initialization'):
                if layer_spec['layer'] in ('linear', 'conv2d'):
                    if layer_spec['activation'] in kaiming_type_activation:
                        nn.init.kaiming_uniform_(module.weight)
                    elif layer_spec['activation'] in xavier_type_activation:
                        nn.init.xavier_uniform_(module.weight)
                elif layer_spec['layer'] in ('lstm', 'gru'):
                    for param in module.parameters():
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param)
                        else:
                            nn.init.normal_(param)

            module_list.append(module)

            # Activation layer
            if activation_name := layer_spec.get('activation'):
                match activation_name:
                    case 'relu':
                        activation = nn.ReLU()
                    case 'elu':
                        activation = nn.ELU()
                    case 'softmax':
                        activation = nn.Softmax(dim=1)
                    case _:
                        raise Exception(f"Invalid activation: {layer_spec['activation']}")
                module_list.append(activation)

        self._layers = nn.ModuleList(module_list)

    def forward(self, *args) -> Tensor | tuple:
        x = args
        for layer in self._layers:
            if isinstance(x, tuple):
                if isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)):
                    x = layer(*x)
                else:
                    x = layer(x[0])
            else:
                x = layer(x)
        return x