import numpy as np
import tensorflow as tf
import utils.logger as logger

tf.get_logger().setLevel('WARNING')

def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
    """
    Create an instance of a single RNN cell.

    Args:
        unit_type (str): Type of the RNN cell (e.g., 'lstm', 'gru').
        num_units (int): Number of units in the RNN cell.
        forget_bias (float): Bias added to forget gates.
        dropout (float): Dropout rate applied to the input of the cell (1 - keep_prob).
        mode (str): Operational mode (train or infer).
        residual_connection (bool): If True, add residual connections.
        device_str (str): Device on which the cell should be placed.
        residual_fn (function): A function to apply as the residual function.

    Returns:
        tf.nn.rnn_cell.RNNCell: Configured RNN cell.
    """
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    if device_str:
        with tf.device(device_str):
            single_cell = _create_cell(unit_type, num_units, forget_bias)
    else:
        single_cell = _create_cell(unit_type, num_units, forget_bias)

    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(
            single_cell, residual_fn=residual_fn)

    return single_cell

def _create_cell(unit_type, num_units, forget_bias):
    """Helper function to create a specific type of RNN cell."""
    if unit_type == "lstm":
        return tf.contrib.rnn.BasicLSTMCell(
            num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
        return tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        return tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units, forget_bias=forget_bias, layer_norm=True)
    elif unit_type == "nas":
        return tf.contrib.rnn.NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
    """
    Create a list of RNN cells.

    Args:
        unit_type (str): Type of the RNN cell.
        num_units (int): Number of units per RNN cell.
        num_layers (int): Total number of layers.
        num_residual_layers (int): Number of residual layers starting from the top.
        forget_bias (float): Bias added to the forget gate of LSTM cells.
        dropout (float): Dropout rate to apply (if applicable).
        mode (str): Operation mode (TRAIN or INFER).
        num_gpus (int): Number of GPUs available.
        base_gpu (int): Index of the first GPU to use for device placement.
        single_cell_fn (function): Function to create a single RNN cell.
        residual_fn (function): Optional function to apply as a residual connection.

    Returns:
        list of tf.nn.rnn_cell.RNNCell: List of configured RNN cells.
    """
    if not single_cell_fn:
        single_cell_fn = _single_cell

    cell_list = []
    for i in range(num_layers):
        # Determine the device for each cell to be placed on
        device_str = None
        if num_gpus > 0:
            gpu_id = (base_gpu + i) % num_gpus
            device_str = '/device:GPU:{}'.format(gpu_id)

        with tf.device(device_str):
            single_cell = single_cell_fn(
                unit_type=unit_type,
                num_units=num_units,
                forget_bias=forget_bias,
                dropout=dropout,
                mode=mode,
                residual_connection=(i >= num_layers - num_residual_layers),
                residual_fn=residual_fn,
                device_str=device_str
            )
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):
    """
    Creates a single or multi-layer RNN cell.

    Args:
        unit_type (str): Type of the RNN cell (e.g., 'lstm', 'gru').
        num_units (int): Number of units in each RNN cell.
        num_layers (int): Total number of layers.
        num_residual_layers (int): Number of residual layers starting from the top.
        forget_bias (float): Bias added to forget gates in LSTM cells.
        dropout (float): Dropout rate (1 - keep_prob).
        mode (str): Operational mode ('TRAIN' or 'INFER').
        num_gpus (int): Number of GPUs available.
        base_gpu (int): Index of the first GPU to use.
        single_cell_fn (function): Function to create a single RNN cell, default is None which uses a built-in single cell creator.

    Returns:
        tf.nn.rnn_cell.RNNCell: Configured RNN cell that may be a single cell or a multi-layer cell.
    """
    # Validate input parameters for common configuration errors
    if num_layers < 1:
        raise ValueError("Number of layers must be at least 1")
    if num_gpus < 0:
        raise ValueError("Number of GPUs cannot be negative")
    if num_units < 1:
        raise ValueError("Number of units per cell must be at least 1")

    # Create a list of RNN cells
    cell_list = _cell_list(
        unit_type=unit_type,
        num_units=num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        num_gpus=num_gpus,
        base_gpu=base_gpu,
        single_cell_fn=single_cell_fn
    )

    # Return appropriate RNN cell based on the number of layers
    if len(cell_list) == 1:  # Single layer
        return cell_list[0]
    else:  # Multiple layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)
