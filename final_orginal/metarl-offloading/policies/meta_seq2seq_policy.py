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

tf.get_logger().setLevel('WARNING')

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.contrib.distributions import Categorical


class FixedSequenceLearningSampleEmbeddingHelper(tf.contrib.seq2seq.SampleEmbeddingHelper):
    def __init__(self, sequence_length, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        """
        Initializes the FixedSequenceLearningSampleEmbeddingHelper.

        Args:
            sequence_length: A 1-D int32 tensor of shape [batch_size] containing the lengths of each sequence.
            embedding: A callable that takes a vector tensor of ids (argmax ids).
            start_tokens: A int32 tensor of shape [batch_size] containing the start tokens.
            end_token: An int32 scalar tensor representing the end token.
            softmax_temperature: Optional float scalar, temperature to apply at sampling time.
            seed: Optional int scalar, the seed for sampling.
        """
        super(FixedSequenceLearningSampleEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )

        # Convert sequence_length to a tensor and check its shape
        self._sequence_length = ops.convert_to_tensor(sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape())

    def sample(self, time, outputs, state, name=None):
        """
        Sample for SampleEmbeddingHelper.

        Args:
            time: A scalar int32 tensor, the current time step.
            outputs: A tensor, the RNN outputs at the current time step.
            state: The RNN state.
            name: An optional string, the name for this operation.

        Returns:
            sample_ids: A tensor containing the sampled ids.
        """
        del time, state  # Unused

        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))

        logits = outputs if self._softmax_temperature is None else outputs / self._softmax_temperature
        sample_id_sampler = Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self._seed)

        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """
        next_inputs_fn for SampleEmbeddingHelper.

        Args:
            time: A scalar int32 tensor, the current time step.
            outputs: A tensor, the RNN outputs at the current time step.
            state: The RNN state.
            sample_ids: A tensor, the sampled ids.
            name: An optional string, the name for this operation.

        Returns:
            finished: A boolean tensor indicating which sequences have finished.
            next_inputs: The next inputs to the RNN.
            state: The RNN state.
        """
        del outputs  # Unused

        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = math_ops.reduce_all(finished)

        next_inputs = control_flow_ops.cond(
            all_finished,
            lambda: self._start_inputs,  # If all sequences are finished, use start inputs
            lambda: self._embedding_fn(sample_ids)  # Otherwise, use the embedding of the sampled ids
        )

        return finished, next_inputs, state


class Seq2SeqNetwork:
    def __init__(self, name,
                 hparams,
                 encoder_inputs,
                 decoder_inputs,
                 decoder_full_length,
                 decoder_targets):
        self.hparams = hparams
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.decoder_full_length = decoder_full_length
        with tf.compat.v1.variable_scope(name, reuse=self.hparams.reuse, initializer=tf.glorot_normal_initializer()):
            self.scope = tf.compat.v1.get_variable_scope().name
            self.embeddings = tf.Variable(tf.random.uniform(
                [self.hparams.n_features,
                 self.hparams.encoder_units],
                -1.0, 1.0), dtype=tf.float32)
            self.encoder_embeddings = tf.contrib.layers.fully_connected(self.encoder_inputs,
                                                                        self.hparams.encoder_units,
                                                                        activation_fn=None,
                                                                        scope="encoder_embeddings",
                                                                        reuse=tf.compat.v1.AUTO_REUSE)
            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings,
                                                             self.decoder_inputs)
            self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets,
                                                         self.hparams.n_features,
                                                         dtype=tf.float32)
            self.output_layer = tf.compat.v1.layers.Dense(self.hparams.n_features, use_bias=False, name="output_projection")
            self.encoder_outputs, self.encoder_state = self.create_encoder(hparams)
            self.decoder_outputs, self.decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="train")
            self.decoder_logits = self.decoder_outputs.rnn_output
            self.pi = tf.nn.softmax(self.decoder_logits)
            self.q = tf.compat.v1.layers.dense(self.decoder_logits, self.hparams.n_features, activation=None,
                                               reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")
            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)
            self.decoder_prediction = self.decoder_outputs.sample_id
            self.sample_decoder_outputs, self.sample_decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                                         self.encoder_state,
                                                                                         model="sample")
            self.sample_decoder_logits = self.sample_decoder_outputs.rnn_output
            self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
            self.sample_q = tf.compat.v1.layers.dense(self.sample_decoder_logits, self.hparams.n_features,
                                                      activation=None, reuse=tf.compat.v1.AUTO_REUSE,
                                                      name="qvalue_layer")
            self.sample_vf = tf.reduce_sum(self.sample_pi * self.sample_q, axis=-1)
            self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id


    def create_encoder(self, hparams):
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            encoder_cell = self.create_rnn_cell(hparams=hparams,)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                sequence_length = None,
                inputs=self.encoder_embeddings,
                dtype=tf.float32,
                time_major=False,
                swap_memory=True,
                scope=scope
            )

        return encoder_outputs, encoder_state

    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE) as decoder_scope:
            if model == "sample":
                helper = FixedSequenceLearningSampleEmbeddingHelper(
                    sequence_length=self.decoder_full_length,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.hparams.start_token),
                    end_token=self.hparams.end_token
                )
            elif model == "train":
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=False)
            decoder_cell = self.create_rnn_cell(hparams=hparams)
            attention_states = encoder_outputs
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.hparams.decoder_hidden_unit, attention_states)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=self.hparams.decoder_hidden_unit)
            decoder_initial_state = (decoder_cell.zero_state(
                tf.size(self.decoder_full_length), dtype=tf.float32).clone(
                cell_state=encoder_state))
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=self.output_layer)
            outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                       output_time_major=False,
                                                                       maximum_iterations=self.decoder_full_length[0])
        return outputs, last_state
    def _single_cell(self, hparams, residual_connection=False, residual_fn=None):
        """Create an instance of a single RNN cell."""
        # dropout (= 1 - keep_prob) is set to 0 during eval and infer
        dropout = hparams.dropout if hparams.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        # Cell Type
        if hparams.unit_type == "lstm":
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                hparams.num_units,
                forget_bias=hparams.forget_bias)
        elif hparams.unit_type == "gru":
            single_cell = tf.contrib.rnn.GRUCell(hparams.num_units)
        elif hparams.unit_type == "layer_norm_lstm":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                hparams.num_units,
                forget_bias=hparams.forget_bias,
                layer_norm=True)
        elif hparams.unit_type == "nas":
            single_cell = tf.contrib.rnn.NASCell(hparams.num_units)
        else:
            raise ValueError("Unknown unit type %s!" % hparams.unit_type)

        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))

        # Residual
        if residual_connection:
            single_cell = tf.contrib.rnn.ResidualWrapper(
                single_cell, residual_fn=residual_fn)

        return single_cell


    def create_rnn_cell(self, hparams, residual_fn=None, single_cell_fn=None):
        if not hparams.single_cell_fn:
            single_cell_fn = self._single_cell

        cell_list = []
        for i in range(hparams.num_layers):
            single_cell = single_cell_fn(
                hparams,
                residual_connection=(i >= hparams.num_layers - hparams.num_residual_layers),
                residual_fn=residual_fn
            )
            cell_list.append(single_cell)
        if len(cell_list) == 1:
            return cell_list[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cell_list)
        
    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class Seq2SeqPolicy():
    def __init__(self, obs_dim, vocab_size, hparams, name="pi"):
        self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32,
                                                        name="decoder_targets_ph_" + name)
        self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32,
                                                       name="decoder_inputs_ph" + name)
        self.obs = tf.compat.v1.placeholder(shape=[None, None, obs_dim], dtype=tf.float32, name="obs_ph" + name)
        self.decoder_full_length = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32,
                                                            name="decoder_full_length" + name)
        self.action_dim = vocab_size
        self.name = name
        self.network = Seq2SeqNetwork(hparams=hparams,
                                      encoder_inputs=self.obs,
                                      decoder_inputs=self.decoder_inputs,
                                      decoder_full_length=self.decoder_full_length,
                                      decoder_targets=self.decoder_targets, name=name)
        self.vf = self.network.vf

        self._dist = CategoricalPd(vocab_size)

    def get_actions(self, observations):
        sess = tf.compat.v1.get_default_session()

        decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

        actions, logits, v_value = sess.run([self.network.sample_decoder_prediction,
                                             self.network.sample_decoder_logits,
                                             self.network.sample_vf],
                                            feed_dict={self.obs: observations,
                                                       self.decoder_full_length: decoder_full_length})

        return actions, logits, v_value

    @property
    def distribution(self):
        return self._dist

    def get_variables(self):
        return self.network.get_variables()

    def get_trainable_variables(self):
        return self.network.get_trainable_variables()

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
    def __init__(self, meta_batch_size, obs_dim, vocab_size, hparams):
        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        self.core_policy = Seq2SeqPolicy(obs_dim, vocab_size, hparams=hparams, name='core_policy')
        self.meta_policies = []
        self.assign_old_eq_new_tasks = []
        for i in range(meta_batch_size):
            self.meta_policies.append(
                Seq2SeqPolicy(obs_dim, vocab_size, hparams=hparams, name="task_" + str(i) + "_policy"))
            self.assign_old_eq_new_tasks.append(
                U.function([], [], updates=[tf.compat.v1.assign(oldv, newv)
                                            for (oldv, newv) in
                                            zipsame(self.meta_policies[i].get_variables(),
                                                    self.core_policy.get_variables())])
            )
        self._dist = CategoricalPd(vocab_size)

    def get_actions(self, observations):
        assert len(observations) == self.meta_batch_size

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
        # async_parameters.
        for i in range(self.meta_batch_size):
            self.assign_old_eq_new_tasks[i]()

    @property
    def distribution(self):
        return self._dist
