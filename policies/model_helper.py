import numpy as np
import tensorflow as tf
import logging

class ResidualWrapper(tf.keras.layers.Layer):
    def __init__(self, cell, output_transform_dim=None):
        super(ResidualWrapper, self).__init__()
        self.cell = cell
        self.output_transform_dim = output_transform_dim
        if output_transform_dim is not None:
            self.dense = tf.keras.layers.Dense(output_transform_dim)
        else:
            self.dense = None

    def call(self, inputs, states, training=None):
        output, new_states = self.cell(inputs, states, training=training)
        if self.dense is not None:
            output = self.dense(output)
        if inputs.shape[-1] != output.shape[-1]:
            # Transform output dimension to match input dimension
            output = tf.keras.layers.Dense(inputs.shape[-1])(output)
        return inputs + output, new_states

    @property
    def state_size(self):
        return self.cell.state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.cell.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)

class AttentionWrapper(tf.keras.layers.Layer):
    def __init__(self, cell, attention_mechanism):
        super(AttentionWrapper, self).__init__()
        self.cell = cell
        self.attention_mechanism = attention_mechanism

    def call(self, inputs, states, training=None):
        query = states[0]  # Assuming the hidden state of the RNN is used as the query
        context_vector, attention_weights = self.attention_mechanism([query, inputs], return_attention_scores=True)
        
        combined_input = tf.concat([inputs, context_vector], axis=-1)
        output, new_states = self.cell(combined_input, states, training=training)
        return output, new_states

    @property
    def state_size(self):
        return self.cell.state_size


def _single_cell(unit_type, num_units, dropout, mode, attention_mechanism, forget_bias=1.0, residual_connection=False, device_str=None, residual_fn=None):

    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == tf.keras.backend.learning_phase() else 0.0

    # Cell Type
    if unit_type == "lstm":
        single_cell = tf.keras.layers.LSTMCell(num_units)
    elif unit_type == "gru":
        single_cell = tf.keras.layers.GRUCell(num_units, dropout=dropout, recurrent_dropout=0.0)
    elif unit_type == "simple_rnn":
        single_cell = tf.keras.layers.SimpleRNNCell(num_units, dropout=dropout, recurrent_dropout=0.0)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.keras.layers.LayerNormalization(tf.keras.layers.LSTMCell(num_units, dropout=dropout, recurrent_dropout=0.0))
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Residual
    if residual_connection:
        single_cell = ResidualWrapper(single_cell)

    # Add attention mechanism if provided
    if attention_mechanism:
        single_cell = AttentionWrapper(single_cell, attention_mechanism)

    # Device Assignment
    if device_str:
        with tf.device(device_str):
            single_cell = tf.keras.layers.StackedRNNCells([single_cell])

    return single_cell

def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0, 
               single_cell_fn=None, residual_fn=None, is_attention=False):
    if not single_cell_fn:
        single_cell_fn = _single_cell

    cell_list = []
    if is_attention :
        attention_mechanism = tf.keras.layers.Attention()

    for i in range(num_layers):
        single_cell = single_cell_fn(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers),
            residual_fn=residual_fn,
            attention_mechanism=attention_mechanism if i == num_layers - 1 else None  # Add attention to the last layer
        )
        cell_list.append(single_cell)

    return cell_list

def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None, is_attention=False):

    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           num_gpus=num_gpus,
                           base_gpu=base_gpu,
                           single_cell_fn=single_cell_fn,
                           is_attention=is_attention)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.keras.layers.StackedRNNCells(cell_list)