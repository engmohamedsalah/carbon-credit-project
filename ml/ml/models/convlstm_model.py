# ml/models/convlstm_model.py

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell implementation."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Args:
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int): Number of channels of hidden state.
            kernel_size (int or tuple): Size of the convolutional kernel.
            bias (bool): Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolution layer for combined inputs and hidden state
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # 4 gates: input, forget, output, cell
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward pass for the ConvLSTM cell.
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, input_dim, height, width)
            cur_state (tuple): Tuple containing the previous hidden state (h_cur) and cell state (c_cur)
                               each of shape (batch_size, hidden_dim, height, width)
        Returns:
            tuple: Next hidden state (h_next) and cell state (c_next)
        """
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Apply convolution
        combined_conv = self.conv(combined)

        # Split the convolutional output into 4 parts for the gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Apply activation functions
        i = torch.sigmoid(cc_i) # Input gate
        f = torch.sigmoid(cc_f) # Forget gate
        o = torch.sigmoid(cc_o) # Output gate
        g = torch.tanh(cc_g)    # Cell gate (candidate cell state)

        # Calculate next cell state
        c_next = f * c_cur + i * g
        # Calculate next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize hidden state and cell state with zeros.
        Args:
            batch_size (int): Batch size.
            image_size (tuple): Height and width of the image (height, width).
        Returns:
            tuple: Initial hidden state and cell state.
        """
        height, width = image_size
        # Use device of the conv layer's parameters
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM network implementation.
    Can handle multiple layers.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim, batch_first=True, bias=True):
        """
        Initialize ConvLSTM network.
        Args:
            input_dim (int): Number of channels in input images.
            hidden_dim (list): List containing the number of channels in the hidden state for each layer.
            kernel_size (list): List containing the kernel size for each layer.
            num_layers (int): Number of ConvLSTM layers.
            output_dim (int): Number of channels in the final output (e.g., number of classes for segmentation).
            batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, channel, height, width).
            bias (bool): Whether or not to add the bias in ConvLSTM cells.
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        # ModuleList holds submodules in a list
        self.cell_list = nn.ModuleList(cell_list)

        # Final convolution layer to map the last hidden state to the desired output dimension
        # Takes the hidden state of the last layer as input
        self.final_conv = nn.Conv2d(in_channels=self.hidden_dim[-1],
                                    out_channels=self.output_dim,
                                    kernel_size=1, # 1x1 convolution for pixel-wise prediction
                                    padding=0,
                                    bias=True)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass for the ConvLSTM network.
        Args:
            input_tensor (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, input_dim, height, width) if batch_first=True
                                        or (seq_len, batch_size, input_dim, height, width) if batch_first=False.
            hidden_state (list, optional): List of initial hidden states for each layer. Defaults to None (initialized as zeros).
        Returns:
            torch.Tensor: Output tensor (prediction for the last time step) of shape (batch_size, output_dim, height, width).
            list: List of tuples containing the last hidden state and cell state for each layer.
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        # Iterate through layers
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            # Iterate through time steps
            for t in range(seq_len):
                # Get input for current time step
                input_t = cur_layer_input[:, t, :, :, :]
                # Update hidden and cell state
                h, c = self.cell_list[layer_idx](input_tensor=input_t, cur_state=[h, c])
                # Store hidden state for this time step
                output_inner.append(h)

            # Stack hidden states for the current layer along the time dimension
            layer_output = torch.stack(output_inner, dim=1)
            # Input for the next layer is the output of the current layer
            cur_layer_input = layer_output

            # Store the final hidden and cell state for this layer
            last_state_list.append([h, c])

        # Use the hidden state of the last layer at the last time step for final prediction
        last_layer_last_hidden_state = layer_output[:, -1, :, :, :]

        # Apply final 1x1 convolution
        final_output = self.final_conv(last_layer_last_hidden_state)

        return final_output, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """Ensure kernel_size is specified correctly."""
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """Extend single layer parameter to multiple layers."""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# Example Usage:
if __name__ == '__main__':
    # Example parameters
    input_channels = 4 # e.g., R, G, B, NIR
    hidden_channels = [64, 64, 128] # Hidden channels per layer
    kernel_sizes = [(3, 3), (3, 3), (3, 3)] # Kernel size per layer
    num_lstm_layers = 3
    output_channels = 1 # e.g., Binary segmentation mask (forest/non-forest or change/no-change)
    batch_size = 4
    seq_length = 5 # Number of time steps in the sequence
    height, width = 64, 64

    # Create model
    model = ConvLSTM(input_dim=input_channels,
                     hidden_dim=hidden_channels,
                     kernel_size=kernel_sizes,
                     num_layers=num_lstm_layers,
                     output_dim=output_channels,
                     batch_first=True,
                     bias=True)

    # Example input tensor (batch, seq_len, channels, height, width)
    input_seq = torch.randn(batch_size, seq_length, input_channels, height, width)

    # Check if CUDA is available
    if torch.cuda.is_available():
        model = model.cuda()
        input_seq = input_seq.cuda()

    # Forward pass
    # The output `last_step_output` will be the prediction for the last time step
    last_step_output, last_states = model(input_seq)

    print("Input sequence shape:", input_seq.shape)
    print("Output shape (last time step prediction):", last_step_output.shape)
    # Example: Output shape: torch.Size([4, 1, 64, 64])

    # Check shapes of last hidden/cell states for each layer
    for i, (h, c) in enumerate(last_states):
        print(f"Layer {i+1} - Last Hidden State Shape: {h.shape}")
        print(f"Layer {i+1} - Last Cell State Shape: {c.shape}")
        # Example: Layer 1 - Last Hidden State Shape: torch.Size([4, 64, 64, 64])
        # Example: Layer 1 - Last Cell State Shape: torch.Size([4, 64, 64, 64])

