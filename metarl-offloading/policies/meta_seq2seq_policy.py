import os
import joblib

import numpy as np
import tensorflow as tf
import policies.model_helper as model_helper

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions import categorical
from policies.distributions.categorical_pd import CategoricalPd
import utils as U
from utils.utils import zipsame
import sys

tf.get_logger().setLevel('WARNING')

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.contrib.distributions import Categorical

class FixedSequenceLearningSampleEmbeddingHelper(tf.contrib.seq2seq.SampleEmbeddingHelper):
    def __init__(self, sequence_length, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        super(FixedSequenceLearningSampleEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )
        self._sequence_length = ops.convert_to_tensor(sequence_length, dtype=tf.int32, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError("Expected sequence_length to be a vector, but received shape: %s" % self._sequence_length.get_shape())
        
        print("Initialization successful: Sequence length set.")

    def sample(self, time, outputs, state, name=None):
        try:
            logits = outputs / self._softmax_temperature if self._softmax_temperature else outputs
            sample_id_sampler = Categorical(logits=logits)
            sample_ids = sample_id_sampler.sample(seed=self._seed)
            return sample_ids
        except Exception as e:
            print("Error during sampling:", e)
            raise

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        try:
            next_time = time + 1
            finished = next_time >= self._sequence_length
            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished,
                lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))
            return finished, next_inputs, state
        except Exception as e:
            print("Error processing next inputs:", e)
            raise



class Seq2SeqNetwork():
    def __init__(self, name, hparams, reuse, encoder_inputs, decoder_inputs, decoder_full_length, decoder_targets):
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        self.reuse = reuse

        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention
        self.unit_type = hparams.unit_type

        self.mode = tf.contrib.learn.ModeKeys.TRAIN
        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers
        self.single_cell_fn = None
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.decoder_full_length = decoder_full_length

        with tf.compat.v1.variable_scope(name, reuse=self.reuse, initializer=tf.glorot_normal_initializer()):
            self.scope = tf.compat.v1.get_variable_scope().name
            self.embeddings = tf.get_variable("embeddings", [self.n_features, self.encoder_hidden_unit],
                                              initializer=tf.random_uniform_initializer(-1.0, 1.0))

            self.encoder_embeddings = tf.contrib.layers.fully_connected(
                inputs=self.encoder_inputs,
                num_outputs=self.encoder_hidden_unit,
                activation_fn=tf.tanh,
                scope="encoder_embeddings",
                reuse=tf.compat.v1.AUTO_REUSE
            )

            self.output_layer = tf.compat.v1.layers.Dense(self.n_features, use_bias=False, name="output_projection")

            if self.is_bidencoder:
                self.encoder_outputs, self.encoder_state = self.create_bidirectional_encoder(hparams)
            else:
                self.encoder_outputs, self.encoder_state = self.create_encoder(hparams)

            # Make sure to use tf.AUTO_REUSE to handle variable reuse properly
            with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):
                self.sample_decoder_outputs, self.sample_decoder_state = self.create_decoder(hparams, self.encoder_outputs, self.encoder_state, model="sample")
                self.sample_decoder_logits = self.sample_decoder_outputs.rnn_output

                with tf.compat.v1.variable_scope("qvalue_layer", reuse=tf.compat.v1.AUTO_REUSE):
                    self.sample_q = tf.compat.v1.layers.dense(self.sample_decoder_logits, self.n_features, activation=tf.tanh, name="kernel")
                
                self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
                self.sample_vf = tf.reduce_sum(input_tensor=self.sample_pi * self.sample_q, axis=-1)
                self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id
                self.sample_decoder_embeddings = tf.one_hot(indices=self.sample_decoder_prediction, depth=self.n_features, dtype=tf.float32)
                self.sample_neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.sample_decoder_embeddings, logits=self.sample_decoder_logits)
                self.loss = tf.reduce_mean(input_tensor=self.sample_neglogp)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hparams.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.compat.v1.train.get_global_step())


    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.encoder_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def _build_decoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        """Build a multi-layer RNN cell that can be used by decoder"""
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.decoder_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def create_encoder(self, hparams, sequence_lengths=None):
        """
        Build and run an RNN encoder based on specified hyperparameters.

        Args:
            hparams: Hyperparameters for building the encoder.
            sequence_lengths: Optional. An array of sequence lengths for each input in the batch,
                            used to dynamically manage RNN unrolling.

        Returns:
            tuple: Tuple containing encoder outputs and encoder state.
        """
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            encoder_cell = self._build_encoder_cell(
                hparams=hparams,
                num_layers=self.num_layers,
                num_residual_layers=self.num_residual_layers
            )
            
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                sequence_length=sequence_lengths,
                inputs=self.encoder_embeddings,
                dtype=tf.float32,
                time_major=self.time_major,
                swap_memory=True,
                scope=scope
            )

        return encoder_outputs, encoder_state


    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE) as decoder_scope:
            helper = None  # Default to None to handle cases where model is not 'sample'
            if model == "sample":
                helper = FixedSequenceLearningSampleEmbeddingHelper(
                    sequence_length=self.decoder_full_length,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
                )

            if self.is_attention:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)
                attention_states = tf.transpose(encoder_outputs, [1, 0, 2]) if self.time_major else encoder_outputs
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.decoder_hidden_unit, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.decoder_hidden_unit)

                decoder_initial_state = decoder_cell.zero_state(tf.size(self.decoder_full_length), dtype=tf.float32).clone(
                    cell_state=encoder_state)
            else:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)
                decoder_initial_state = encoder_state

            # Ensure helper is initialized before creating the decoder
            if not helper:
                raise ValueError("Helper not defined for the decoder.")

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=self.output_layer)

            outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=self.time_major,
                maximum_iterations=self.decoder_full_length[0])
            
            return outputs, last_state

    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

class Seq2SeqPolicy():
    def __init__(self, obs_dim, encoder_units, decoder_units, vocab_size, learning_rate, name="pi"):
        """
        Initialize the Seq2SeqPolicy class with optional training capabilities.

        Args:
            obs_dim (int): The dimensionality of the observation space.
            encoder_units (int): Number of units in each encoder cell.
            decoder_units (int): Number of units in each decoder cell.
            vocab_size (int): Size of the vocabulary.
            enable_training (bool): If True, includes training capabilities.
            name (str): Identifier for placeholders and operations.
        """
        self.obs = tf.compat.v1.placeholder(shape=[None, None, obs_dim], dtype=tf.float32, name="obs_ph_" + name)
        self.decoder_full_length = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="decoder_full_length_" + name)
        self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs_ph_" + name)
        self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_targets_ph_" + name)

        self.hparams = tf.contrib.training.HParams(
            unit_type="lstm",
            encoder_units=encoder_units,
            decoder_units=decoder_units,
            n_features=vocab_size,
            time_major=False,
            is_attention=True,
            forget_bias=1.0,
            dropout=0,
            num_gpus=1,
            num_layers=2,
            num_residual_layers=0,
            start_token=0,
            end_token=2,
            is_bidencoder=False,
            learning_rate=learning_rate
        )

        self.network = Seq2SeqNetwork(
            hparams=self.hparams,
            reuse=tf.compat.v1.AUTO_REUSE,
            encoder_inputs=self.obs,
            decoder_inputs=self.decoder_inputs,
            decoder_full_length=self.decoder_full_length,
            decoder_targets=self.decoder_targets,
            name=name
        )
        with tf.compat.v1.variable_scope("model", reuse=tf.AUTO_REUSE):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets, logits=self.network.sample_decoder_logits))
            self.train_op = self.optimizer.minimize(self.loss)

    def train(self, session, observations, decoder_inputs, decoder_targets):
        """
        Train the model on a batch of data using an exponentially decaying learning rate.

        Args:
            session (tf.Session): TensorFlow session where the operations will be run.
            observations (np.array): Batch of observations.
            decoder_inputs (np.array): Decoder inputs for training.
            decoder_targets (np.array): Target outputs for the decoder.
            decoder_lengths (np.array): Lengths of each sequence in the batch.

        Returns:
            float: Loss value for the training batch.
        """
        decoder_lengths = np.array( [observations.shape[1]] * observations.shape[0] , dtype=np.int32)
        print(decoder_lengths)
        # Ensure global step and learning rate are initialized here if not already defined in __init__
        if not hasattr(self, 'global_step'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(
                self.hparams.learning_rate,  # Initial learning rate
                self.global_step,               # Current index into the dataset
                100,                            # Decay step
                0.96,                           # Decay rate
                staircase=True)                 # Whether to apply decay in a discrete staircase, as opposed to smoothly

        # Initialize the optimizer within the train function scope if not already part of the class init
        if not hasattr(self, 'optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # Execute the training operation along with the loss computation in a single session run
        _, loss_val = session.run([self.train_op, self.loss], feed_dict={
            self.obs: observations,
            self.decoder_inputs: decoder_inputs,
            self.decoder_targets: decoder_targets,
            self.decoder_full_length: decoder_lengths
        })

        return loss_val


    def get_actions(self, observations):
        session = tf.compat.v1.get_default_session()
        decoder_lengths = np.array( [observations.shape[1]] * observations.shape[0] , dtype=np.int32)
        actions, logits, v_value = session.run(
            [self.network.sample_decoder_prediction, self.network.sample_decoder_logits, self.network.sample_vf],
            feed_dict={self.obs: observations, self.decoder_full_length: decoder_lengths}
        )
        return actions, logits, v_value

    def get_variables(self):
        return self.network.get_variables()

    # def get_trainable_variables(self):
    #     return self.network.get_trainable_variables()

    def save_variables(self, save_path, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        variables = self.get_variables()

        ps = sess.run(variables)
        save_dict = {v.name: value for v, value in zip(variables, ps)}

        dirname = os.path.dirname(save_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)

        joblib.dump(save_dict, save_path)

    def load_variables(self, load_path, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        variables = self.get_variables()

        loaded_params = joblib.load(os.path.expanduser(load_path))
        restores = []

        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                restores.append(v.assign(d))
        else:
            for v in variables:
                restores.append(v.assign(loaded_params[v.name]))

        sess.run(restores)


class MetaSeq2SeqPolicy():
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units, vocab_size, learning_rate):
        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        self.meta_policies = []
        for i in range(meta_batch_size):
            policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, learning_rate,
                                   name="task_" + str(i) + "_policy_" + str(encoder_units) + "_" + str(decoder_units))
            self.meta_policies.append(policy)

    def train(self, observations, session=None, ):
        """
        Train each model on a batch of data corresponding to each task.

        Args:
            session (tf.Session): TensorFlow session where the operations will be run.
            meta_observations (list of np.array): List of observation batches, one per task.
            meta_decoder_inputs (list of np.array): List of decoder input batches, one per task.
            meta_decoder_targets (list of np.array): List of decoder target batches, one per task.

        Returns:
            list of float: List of loss values for each task.
        """
        # assert len(meta_observations) == self.meta_batch_size
        # assert len(meta_decoder_inputs) == self.meta_batch_size
        # assert len(meta_decoder_targets) == self.meta_batch_size
        session = session or tf.compat.v1.get_default_session()
        meta_losses = []
        for i, obser_per_task in enumerate(observations):
            observations = obser_per_task
            decoder_inputs = session.run(self.meta_policies[i].decoder_inputs)
            decoder_targets = self.meta_policies[i].decoder_targets
            decoder_lengths = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

            loss_val = self.meta_policies[i].train(session, observations, decoder_inputs, decoder_targets)
            meta_losses.append(loss_val)

        return meta_losses

    def get_actions(self, observations):
        assert len(observations) == self.meta_batch_size
        meta_actions, meta_logits, meta_v_values = [], [], []
        for i, obser_per_task in enumerate(observations):
            action, logits, v_value = self.meta_policies[i].get_actions(obser_per_task)
            meta_actions.append(np.array(action))
            meta_logits.append(np.array(logits))
            meta_v_values.append(np.array(v_value))
        return meta_actions, meta_logits, meta_v_values
