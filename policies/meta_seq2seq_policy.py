import os
import joblib
import numpy as np
import tensorflow as tf
import logging
import re

# TensorFlow 2.x compatible imports
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Custom imports (ensure these are compatible with TensorFlow 2.x)
import policies.model_helper as model_helper
import utils as U
from utils.utils import zipsame
from policies.distributions.categorical_pd import CategoricalPd

# Set logging level
logging.basicConfig(level=logging.WARNING)
from tensorflow.keras.layers import Embedding, Dense, LSTM, GRU, RNN, LSTMCell, GRUCell, Attention
from tensorflow.keras import Model
import policies.model_helper as model_helper

class FixedSequenceLearningSampleEmbeddingHelper(tf.keras.layers.Layer):
    def __init__(self, embedding, start_tokens, end_token, sequence_length, softmax_temperature=None, seed=None, **kwargs):
        super(FixedSequenceLearningSampleEmbeddingHelper, self).__init__(**kwargs)
        self.embedding = embedding
        self.start_tokens = start_tokens
        self.end_token = end_token
        self.sequence_length = sequence_length
        self.softmax_temperature = softmax_temperature
        self.seed = seed

    def initialize(self, inputs):
        finished = tf.tile([False], [tf.shape(inputs)[0]])
        start_inputs = self.embedding(self.start_tokens)
        return finished, start_inputs

    def sample(self, time, outputs, state):
        sample_ids = tf.random.categorical(outputs, 1, dtype=tf.int32, seed=self.seed)
        return tf.squeeze(sample_ids, axis=-1)

    def next_inputs(self, time, outputs, state, sample_ids):
        finished = (time + 1 >= self.sequence_length)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            lambda: self.embedding(self.start_tokens),
            lambda: self.embedding(sample_ids))
        return finished, next_inputs, state

class Seq2SeqNetwork:
    def __init__(self, name, hparams,  encoder_inputs, decoder_inputs, decoder_full_length, decoder_targets):
        logging.debug('Start Seq2SeqNetwork')
        self.encoder_hidden_unit = hparams['encoder_units']
        self.decoder_hidden_unit = hparams['decoder_units']
        self.is_bidencoder = hparams['is_bidencoder']

        self.n_features = hparams['n_features']
        self.time_major = hparams['time_major']
        self.is_attention = hparams['is_attention']

        self.unit_type = hparams['unit_type']
        self.mode = tf.estimator.ModeKeys.TRAIN

        self.num_layers = hparams['num_layers']
        self.num_residual_layers = hparams['num_residual_layers']

        self.single_cell_fn = None
        self.start_token = hparams['start_token']
        self.end_token = hparams['end_token']

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

        self.decoder_full_length = decoder_full_length

        self.embeddings = tf.Variable(tf.random.uniform([self.n_features, self.encoder_hidden_unit], -1.0, 1.0), dtype=tf.float32)
        self.encoder_embeddings = Dense(self.encoder_hidden_unit, activation=None, name="encoder_embeddings")(self.encoder_inputs)
        if isinstance(self.decoder_inputs, tf.TensorArray):
            self.decoder_inputs = self.decoder_inputs.stack()
        self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
        if isinstance(self.decoder_targets, tf.TensorArray):
            self.decoder_targets = self.decoder_targets.stack()
        self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets, self.n_features, dtype=tf.float32)
        self.output_layer = Dense(self.n_features, use_bias=False, name="output_projection")

        self.encoder_outputs, self.encoder_state = self.create_encoder(hparams)

        self.decoder_outputs, self.decoder_state = self.create_decoder(hparams, self.encoder_outputs, self.encoder_state, model="train")
        self.decoder_logits = self.decoder_outputs.rnn_output
        self.pi = tf.nn.softmax(self.decoder_logits)
        self.q = Dense(self.n_features, activation=None, name="qvalue_layer")(self.decoder_logits)
        self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)
        self.decoder_prediction = self.decoder_outputs.sample_id

        # Sample decoder
        self.sample_decoder_outputs, self.sample_decoder_state = self.create_decoder(hparams, self.encoder_outputs, self.encoder_state, model="sample")
        self.sample_decoder_logits = self.sample_decoder_outputs.rnn_output
        self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
        self.sample_q = Dense(self.n_features, activation=None, name="qvalue_layer")(self.sample_decoder_logits)
        self.sample_vf = tf.reduce_sum(self.sample_pi * self.sample_q, axis=-1)
        self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id

        # Greedy decoder
        self.greedy_decoder_outputs, self.greedy_decoder_state = self.create_decoder(hparams, self.encoder_outputs, self.encoder_state, model="greedy")
        self.greedy_decoder_logits = self.greedy_decoder_outputs.rnn_output
        self.greedy_pi = tf.nn.softmax(self.greedy_decoder_logits)
        self.greedy_q = Dense(self.n_features, activation=None, name="qvalue_layer")(self.greedy_decoder_logits)
        self.greedy_vf = tf.reduce_sum(self.greedy_pi * self.greedy_q, axis=-1)
        self.greedy_decoder_prediction = self.greedy_decoder_outputs.sample_id

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        logging.debug('Start _build_encoder_cell')
        return model_helper.create_rnn_cell(
            unit_type=hparams['unit_type'],
            num_units=hparams['encoder_units'],
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams['forget_bias'],
            dropout=hparams['dropout'],
            num_gpus=hparams['num_gpus'],
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn,
            is_attention=hparams['is_attention'])

    def _build_decoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        logging.debug('Start _build_decoder_cell')
        return model_helper.create_rnn_cell(
            unit_type=hparams['unit_type'],
            num_units=hparams['decoder_units'],
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams['forget_bias'],
            dropout=hparams['dropout'],
            num_gpus=hparams['num_gpus'],
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn,
            is_attention=hparams['is_attention'])
    
    # @tf.function
    def create_encoder(self, hparams):
        logging.debug('Start create_encoder')
        encoder_cell = self._build_encoder_cell(hparams=hparams, num_layers=self.num_layers, num_residual_layers=self.num_residual_layers)
        encoder = tf.keras.layers.RNN(encoder_cell, return_sequences=True, return_state=True)
        encoder_outputs_and_states = encoder(self.encoder_embeddings)

        # Unpack the outputs and states
        encoder_outputs = encoder_outputs_and_states[0]
        encoder_states = encoder_outputs_and_states[1:]

        return encoder_outputs, encoder_states

    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        logging.debug('Start create_decoder model = %s, is_attention = %s', str(model), str(self.is_attention))

        # Build the decoder cell
        decoder_cell = self._build_decoder_cell(hparams=hparams, num_layers=self.num_layers, num_residual_layers=self.num_residual_layers)

        # Use the decoder cell in an RNN layer
        decoder = tf.keras.layers.RNN(
            decoder_cell, return_sequences=True, return_state=True, time_major=self.time_major)


        # Use the combined_initial_state in your decoder
        if model == "greedy":
            # For greedy decoding
            decoder = tf.keras.layers.RNN(
                decoder_cell, return_sequences=True, return_state=True, time_major=self.time_major)
            # Implement greedy decoding logic here as per your requirements
        elif model == "sample":
            # For sample decoding
            decoder = tf.keras.layers.RNN(
                decoder_cell, return_sequences=True, return_state=True, time_major=self.time_major)
            # Implement sample decoding logic here as per your requirements
        elif model == "train":
            # For training
            print("Decoder embeddings shape:", self.decoder_embeddings.shape)

            # Ensure the decoder embeddings have the correct shape
            if len(self.decoder_embeddings.shape) != 3:
                raise ValueError("Decoder embeddings should have 3 dimensions [batch_size, timesteps, feature_size]")

            # Proceed with the RNN layer
            outputs, last_state, _ = decoder(
                inputs=self.decoder_embeddings, training=True)
            return outputs, last_state
        else:
            raise ValueError("Unsupported model type: {}".format(model))



    def get_variables(self):
        # This method returns all variables (trainable and non-trainable)
        return [var for var in tf.compat.v1.global_variables() if self.scope in var.name]

    def get_trainable_variables(self):
        # This method returns only the trainable variables
        return [var for var in tf.compat.v1.trainable_variables() if self.scope in var.name]


        # hparams = tf.contrib.training.HParams(
        #     unit_type="lstm", 
        #     encoder_units=encoder_units,
        #     decoder_units=decoder_units,

        #     n_features=vocab_size,
        #     time_major=False,
        #     is_attention=True,
        #     forget_bias=1.0,
        #     dropout=0,
        #     num_gpus=1,
        #     num_layers=2,
        #     num_residual_layers=0,
        #     start_token=0,
        #     end_token=2,
        #     is_bidencoder=False
        # )
class Seq2SeqPolicy:
    def __init__(self, obs_dim, encoder_units, decoder_units, vocab_size, name="pi"):
        logging.debug('Start Seq2SeqPolicy name=%s', name)
        self.decoder_targets = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, name="decoder_targets_ph_" + name)
        self.decoder_inputs = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, name="decoder_inputs_ph_" + name)
        self.obs_dim = obs_dim
        self.obs = tf.keras.layers.Input(shape=(None, obs_dim), dtype=tf.float32, name="obs_input_" + name)
        self.decoder_full_length = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="decoder_full_length_input_" + name)

        self.action_dim = vocab_size
        self.name = name

        hparams = {
            'unit_type': "lstm",
            'encoder_units': 256,
            'decoder_units': 256,
            'n_features': vocab_size,
            'time_major': False,
            'is_attention': True,
            'forget_bias': 1.0,
            'dropout': 0,
            'num_gpus': 1,
            'num_layers': 3,
            'num_residual_layers': 2,
            'start_token': 0,
            'end_token': 2,
            'is_bidencoder': True
        }

        self.network = Seq2SeqNetwork(hparams=hparams, 
                                      encoder_inputs=self.obs,
                                      decoder_inputs=self.decoder_inputs,
                                      decoder_full_length=self.decoder_full_length,
                                      decoder_targets=self.decoder_targets, name=name)

        self.vf = self.network.vf
        self._dist = CategoricalPd(vocab_size)
        logging.debug('END MetaSeq2SeqPolicy')

    def get_actions(self, observations):
        logging.debug('Start Seq2SeqPolicy get_actions')

        decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)
        actions, logits, v_value = self.network.sample_decoder_prediction, self.network.sample_decoder_logits, self.network.sample_vf

        return actions, logits, v_value

    @property
    def distribution(self):
        return self._dist

    def get_variables(self):
        return self.network.get_variables()

    def get_trainable_variables(self):
        return self.network.get_trainable_variables()

    def save_variables(self, save_path):
        variables = self.get_variables()
        ps = [v.numpy() for v in variables]
        save_dict = {v.name: value for v, value in zip(variables, ps)}

        dirname = os.path.dirname(save_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)

        joblib.dump(save_dict, save_path)

    def load_variables(self, load_path):
        variables = self.get_variables()
        loaded_params = joblib.load(os.path.expanduser(load_path))
        restores = []

        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                v.assign(d)
        else:
            for v in variables:
                v.assign(loaded_params[v.name])


class MetaSeq2SeqPolicy:
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units, vocab_size):
        logging.debug('Start MetaSeq2SeqPolicy')
        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size

        self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name='core_policy')
        self.meta_policies = [Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name="task_"+str(i)+"_policy") for i in range(meta_batch_size)]

        self._dist = CategoricalPd(vocab_size)
        logging.debug('END MetaSeq2SeqPolicy')
        self.selected_indices = set()

    def get_actions(self, observations):
        logging.debug('Start MetaSeq2SeqPolicy get_actions')
        meta_actions = []
        meta_logits = []
        meta_v_values = []
        for i, obser_per_task in enumerate(observations):
            action, logits, v_value = self.meta_policies[i].get_actions(obser_per_task)
            meta_actions.append(np.array(action))
            meta_logits.append(np.array(logits))
            meta_v_values.append(np.array(v_value))
        return meta_actions, meta_logits, meta_v_values

    def async_parameters(self):
        logging.info('MetaSeq2SeqPolicy async_parameters')
        for i in range(self.meta_batch_size):
            new_params = self.core_policy.get_trainable_variables()
            self.set_params(new_params, i)

    @property
    def distribution(self):
        return self._dist

    def select_random_policy(self, batch_size):
        all_indices = set(range(1, batch_size))
        available_indices = all_indices - self.selected_indices
        if not available_indices:
            self.selected_indices = set()
            available_indices = all_indices
        random_index = np.random.choice(list(available_indices))
        self.selected_indices.add(random_index)
        return random_index

    def get_params(self, index):
        """Get the current parameters of the policy."""
        variables = self.meta_policies[index].network.get_trainable_variables()
        return {v.name: v.numpy() for v in variables}

    def set_params(self, new_params, index):
        specific_policy = self.meta_policies[index]
        new_prefix = f'task_{index}_policy/'
        updated_params = {new_prefix + re.sub(r'^task_[0-9]{1,2}_policy/', '', key): value for key, value in new_params.items()}
        trainable_vars = {var.name: var for var in specific_policy.network.get_trainable_variables()}

        for key, value in updated_params.items():
            var = trainable_vars.get(key, None)
            if var is not None:
                var.assign(value)
            else:
                raise ValueError(f"Variable {key} not found in specific_policy{index}")

    def set_params_core(self, new_params):
        new_params_prefixed = {'core_policy/' + re.sub(r'^task_[0-9]{1,2}_policy/', '', key): value for key, value in new_params.items()}
        for var in self.core_policy.get_trainable_variables():
            var_name = var.name
            if var_name in new_params_prefixed:
                var.assign(new_params_prefixed[var_name])
            else:
                raise KeyError(f"Variable {var_name} not found in new_params_prefixed. Make sure new_params contains this key.")

