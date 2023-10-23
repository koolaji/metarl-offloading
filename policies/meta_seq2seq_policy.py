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
import logging
import re

tf.get_logger().setLevel('WARNING')


class FixedSequenceLearningSampleEmbedingHelper(tf.contrib.seq2seq.SampleEmbeddingHelper):
    def __init__(self, sequence_length, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        logging.debug('Start FixedSequenceLearningSampleEmbedingHelper')
        super(FixedSequenceLearningSampleEmbedingHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )
        self._sequence_length = ops.convert_to_tensor(
            sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape())

    def sample(self, time, outputs, state, name=None):
        """sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        sample_id_sampler = categorical.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self._seed)

        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del outputs  # unused by next_inputs_fn

        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = math_ops.reduce_all(finished)

        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)


class Seq2SeqNetwork:
    def __init__(self, name,
                 hparams, reuse,
                 encoder_inputs,
                 decoder_inputs,
                 decoder_full_length,
                 decoder_targets):
        logging.debug('Start Seq2SeqNetwork')
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        self.reuse = reuse

        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention

        self.unit_type = hparams.unit_type

        # default setting
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
            self.embeddings = tf.Variable(tf.random.uniform(
                [self.n_features,
                 self.encoder_hidden_unit],
                -1.0, 1.0), dtype=tf.float32)
            # using a fully connected layer as embeddings
            self.encoder_embeddings = tf.contrib.layers.fully_connected(self.encoder_inputs,
                                                                        self.encoder_hidden_unit,
                                                                        activation_fn = None,
                                                                        scope="encoder_embeddings",
                                                                        reuse=tf.compat.v1.AUTO_REUSE)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings,
                                                             self.decoder_inputs)
            self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets,
                                                         self.n_features,
                                                         dtype=tf.float32)
            self.output_layer = tf.compat.v1.layers.Dense(self.n_features, use_bias=False, name="output_projection")
            self.encoder_outputs, self.encoder_state = self.create_encoder(hparams)

            # training decoder
            self.decoder_outputs, self.decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="train")
            self.decoder_logits = self.decoder_outputs.rnn_output
            self.pi = tf.nn.softmax(self.decoder_logits)
            self.q = tf.compat.v1.layers.dense(self.decoder_logits, self.n_features, activation=None,
                                               reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")
            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)
            self.decoder_prediction = self.decoder_outputs.sample_id

            # sample decoder
            self.sample_decoder_outputs, self.sample_decoder_state = \
                self.create_decoder(hparams, self.encoder_outputs, self.encoder_state, model="sample")
            self.sample_decoder_logits = self.sample_decoder_outputs.rnn_output
            self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
            self.sample_q = tf.compat.v1.layers.dense(self.sample_decoder_logits, self.n_features,
                                                      activation=None, reuse=tf.compat.v1.AUTO_REUSE,
                                                      name="qvalue_layer")
            self.sample_vf = tf.reduce_sum(self.sample_pi*self.sample_q, axis=-1)
            self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id

            # greedy decoder
            self.greedy_decoder_outputs, self.greedy_decoder_state = \
                self.create_decoder(hparams, self.encoder_outputs, self.encoder_state, model="greedy")
            self.greedy_decoder_logits = self.greedy_decoder_outputs.rnn_output
            self.greedy_pi = tf.nn.softmax(self.greedy_decoder_logits)
            self.greedy_q = tf.compat.v1.layers.dense(self.greedy_decoder_logits, self.n_features, activation=None,
                                                      reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")
            self.greedy_vf = tf.reduce_sum(self.greedy_pi * self.greedy_q, axis=-1)
            self.greedy_decoder_prediction = self.greedy_decoder_outputs.sample_id

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""
        logging.debug('Start _build_encoder_cell')
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
        logging.debug('Start _build_decoder_cell')
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

    def create_encoder(self, hparams):
        logging.debug('Start create_encoder')
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            encoder_cell = self._build_encoder_cell(hparams=hparams,
                                                    num_layers=self.num_layers,
                                                    num_residual_layers=self.num_residual_layers)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                sequence_length = None,
                inputs=self.encoder_embeddings,
                dtype=tf.float32,
                time_major=self.time_major,
                swap_memory=True,
                scope=scope
            )

        return encoder_outputs, encoder_state
    
    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        logging.debug('Start create_decoder  model = %s, is_attention = %s', str(model), str(self.is_attention))
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE) as decoder_scope:
            if model == "greedy":
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embeddings,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
                )

            elif model == "sample":
                helper = FixedSequenceLearningSampleEmbedingHelper(
                    sequence_length=self.decoder_full_length,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
                )

            elif model == "train":
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)

            if self.is_attention:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)
                if self.time_major:
                    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
                else:
                    attention_states = encoder_outputs

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.decoder_hidden_unit, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.decoder_hidden_unit)

                decoder_initial_state = (
                    decoder_cell.zero_state(tf.size(self.decoder_full_length),
                                            dtype=tf.float32).clone(
                        cell_state=encoder_state))
            else:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)

                decoder_initial_state = encoder_state

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=self.output_layer)

            outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                       output_time_major=self.time_major,
                                                                       maximum_iterations=self.decoder_full_length[0])
        return outputs, last_state

    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        logging.debug('Start Seq2SeqPolicy scope=%s' , self.scope)
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class Seq2SeqPolicy:
    def __init__(self, obs_dim, encoder_units,
                 decoder_units, vocab_size, name="pi"):
        logging.debug('Start Seq2SeqPolicy name=%s' , name)
        self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32,
                                                        name="decoder_targets_ph_"+name)
        self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32,
                                                       name="decoder_inputs_ph"+name)
        self.obs = tf.compat.v1.placeholder(shape=[None, None, obs_dim], dtype=tf.float32, name="obs_ph"+name)
        self.decoder_full_length = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32,
                                                            name="decoder_full_length"+name)

        self.action_dim = vocab_size # number of possible action in our case means two action (local or MEC) 
        self.name = name
        hparams = tf.contrib.training.HParams(
            unit_type="lstm", # layer_norm_lstm or Gru
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
            is_bidencoder=False
        )
        self.network = Seq2SeqNetwork(hparams=hparams, reuse=tf.compat.v1.AUTO_REUSE,
                 encoder_inputs=self.obs,
                 decoder_inputs=self.decoder_inputs,
                 decoder_full_length=self.decoder_full_length,
                 decoder_targets=self.decoder_targets, name = name)

        self.vf = self.network.vf

        self._dist = CategoricalPd(vocab_size)
        logging.debug('END MetaSeq2SeqPolicy')
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())  

    def get_actions(self, observations):
        logging.debug('Start Seq2SeqPolicy get_actions')
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


class MetaSeq2SeqPolicy:
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units,
                 vocab_size):
        logging.debug('Start MetaSeq2SeqPolicy')
        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size

        self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name='core_policy')

        self.meta_policies = []

        self.assign_old_eq_new_tasks = []
        self.selected_indices = set()
        

        for i in range(meta_batch_size):
            self.meta_policies.append(Seq2SeqPolicy(obs_dim, encoder_units, decoder_units,
                                                    vocab_size, name="task_"+str(i)+"_policy"))

            self.assign_old_eq_new_tasks.append(
                U.function([], [], updates=[tf.compat.v1.assign(oldv, newv)
                                            for (oldv, newv) in
                                            zipsame(self.meta_policies[i].get_variables(),
                                                    self.core_policy.get_variables())])
                )

        self._dist = CategoricalPd(vocab_size)
        logging.debug('END MetaSeq2SeqPolicy')


    def get_actions(self, observations):
        logging.debug('Start MetaSeq2SeqPolicy get_actions')
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
    
    # def get_params(self, sess):
    #     """Get the current parameters of the policy."""
    #     # logging.info('get_params ')
    #     # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     variables = self.core_policy.get_trainable_variables()
    #     return {v.name: sess.run(v) for v in variables}

    # def set_params(self, new_params, sess):
    #         """Set the policy parameters to new values."""
    #     #with tf.compat.v1.Session() as sess:
    #         sess = sess or  tf.compat.v1.get_default_session()
    #         sess.run(tf.compat.v1.global_variables_initializer())
    #         for var in self.core_policy.get_trainable_variables():
    #             value = new_params[var.name]
    #             var.load(value, sess)

    #def set_params(self, new_params, sess=None):
    #    """Set the policy parameters to new values."""
    #    sess = sess or  tf.compat.v1.get_default_session()
    #    for var in self.core_policy.get_trainable_variables():
    #        value = new_params[var.name]
    #        sess.run(var.assign(value))

    # def get_random_params(self, sess):
    # #         logging.info('get_random_params ')
    # #     # with tf.compat.v1.Session() as sess:
    # #     #     sess.run(tf.compat.v1.global_variables_initializer())
    # #         params = self.get_params(sess)
    # #         random_params = {}
    # #         # logging.info('get_random_params %s %s', str(len(params)), type(params))
    # #         for key, value in params.items():
    # #             np.random.seed(None)
    # #             random_value = np.random.randn(*value.shape)
    # #             # logging.info(f"Random value for {value.shape} key {key}")
    # #             random_params[key] = random_value            
    # #         return random_params
    #     # Randomly select an index from the list of meta_policies
    #     random_index = np.random.randint(0, len(self.meta_policy.meta_policies))
    #     # Return the randomly selected policy
    #     return self.meta_policy.meta_policies[random_index], random_index

    def set_params(self, new_params, index, sess=None):
        """Set the policy parameters to new values for a specific policy."""
        sess = sess or tf.compat.v1.get_default_session()
        # Get the specific policy using the provided index
        specific_policy = self.meta_policies[index]
        
        # Compile a regular expression pattern to match the prefix
        prefix_pattern = re.compile(r'^task_[0-9]_policy/')
        
        # Loop through each key-value pair in new_params
        for key, value in new_params.items():
            # Remove the prefix from the key to get the variable name
            var_name = prefix_pattern.sub('', key)  # This line replaces the prefix with an empty string
            
            # Locate the variable within the specific policy
            var = next((v for v in specific_policy.network.get_trainable_variables() if prefix_pattern.sub('', v.name) == var_name), None)
            if var is not None:
                # Ensure the shapes match before assignment
                assert var.get_shape() == value.shape, f"Shape mismatch: {var.get_shape()} vs {value.shape}"
                sess.run(var.assign(value))
            else:
                print(f"Variable {var_name} not found in specific_policy")
            
    def select_random_policy(self, batch_size):
        # Get the set of all indices
        all_indices = set(range(1, batch_size))
        # Compute the set of available indices by removing the selected indices from all indices
        available_indices = all_indices - self.selected_indices
        # If all indices have been selected, reset the selected indices set
        if not available_indices:
            self.selected_indices = set()
            available_indices = all_indices
        
        # Now, select a random index from the available indices
        random_index = np.random.choice(list(available_indices))
        # Add the selected index to the selected_indices set
        self.selected_indices.add(random_index)
        
        # Return the randomly selected index
        return random_index
    

    def get_params(self, sess, index):
        """Get the current parameters of the policy."""
        # logging.info('get_params ')
        # with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        variables = self.meta_policies[index].network.get_trainable_variables()
        return {v.name: sess.run(v) for v in variables}

    # def get_random_params(self, policy, sess):
    #     # Assuming get_params is a method that retrieves the current parameters of a policy
    #     current_params = policy.get_params(sess)
    #     random_params = {}
    #     for key, value in current_params.items():
    #         np.random.seed(None)  # Resetting the random seed at each iteration may not be necessary
    #         random_value = np.random.randn(*value.shape)
    #         random_params[key] = random_value
    #     return random_params
    
    # def evaluate_policy(self, policy_index, environment):
    #     """Evaluate the policy with the given index on the provided environment."""
    #     # Get the policy from the meta_policies list using the index
    #     policy = self.meta_policies[policy_index]
        
    #     # Reset the environment to start a new evaluation episode
    #     state = environment.reset()
        
    #     # Initialize a variable to keep track of the cumulative reward
    #     cumulative_reward = 0
        
    #     # Assume the environment has a method `is_done` to check if the episode is over
    #     while not environment.is_done():
    #         # Get the action from the policy (you'll need to implement a method to do this)
    #         action, _, _ = policy.get_actions(state)
            
    #         # Assume the environment has a method `step` to execute the action and return a reward
    #         next_state, reward, _ = environment.step(action)
            
    #         # Update the cumulative reward
    #         cumulative_reward += reward
            
    #         # Update the state for the next iteration
    #         state = next_state
        
    #     return cumulative_reward