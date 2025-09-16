import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

class PINN(tf.keras.Model):
    def __init__(self, num_layers: int, num_neurons: int, input_dim: int, output_dim: int):
        super(PINN, self).__init__()
        self._input_dim: int = input_dim
        self._output_dim: int = output_dim
        self._num_layers: int = num_layers
        self._num_neurons: int = num_neurons
        self._layers_list = [
            Dense(
                self._num_neurons, activation='tanh', kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=l2(0.01)
            ) for _ in range(self._num_layers)
        ]
        self._output_layer = Dense(output_dim)  # Assuming you want an output shape of (None, 3)

    def call(self, inputs, training=None, mask=None):
        for idx, layer in enumerate(self._layers_list):
            if idx == 0:
                x = layer(inputs)
            else:
                x = layer(x)
        outputs = self._output_layer(x)
        return outputs

    def set_num_layers(self, num_layers: int) -> None:
        self._num_layers = num_layers
        self._reset_layers_list()

    def set_num_neurons(self, num_neurons: int) -> None:
        self._num_neurons = num_neurons
        self._reset_layers_list()

    def _reset_layers_list(self) -> None:
        self._layers_list = [
            Dense(
                self._num_neurons, activation='tanh', kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=l2(0.01)
            ) for _ in range(self._num_layers)
        ]

    def __str__(self) -> str:
        return (
            f"inputs: {self._input_dim}\n"
            f"outputs: {self._output_dim}\n"
            f"hidden_layers: {self._num_layers}\n"
            f"neurons_per_hidden_layer: {self._num_neurons}"
        )