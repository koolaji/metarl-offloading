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
from tensorflow.contrib.seq2seq import AttentionWrapper, LuongAttention

class FixedSequenceLearningSampleEmbedingHelper(tf.contrib.seq2seq.SampleEmbeddingHelper):
    def __init__(self, sequence_length, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        super(FixedSequenceLearningSampleEmbedingHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )
        self._sequence_length = ops.convert_to_tensor(
            sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape())

    # def sample(self, time, outputs, state, name=None):
    #     """sample for SampleEmbeddingHelper."""
    #     del time, state  # unused by sample_fn
    #     # Outputs are logits, we sample instead of argmax (greedy).
    #     if not isinstance(outputs, ops.Tensor):
    #         raise TypeError("Expected outputs to be a single Tensor, got: %s" %
    #                         type(outputs))
    #     if self._softmax_temperature is None:
    #         logits = outputs
    #     else:
    #         logits = outputs / self._softmax_temperature

    #     sample_id_sampler = categorical.Categorical(logits=logits)
    #     sample_ids = sample_id_sampler.sample(seed=self._seed)

    #     return sample_ids

    # def next_inputs(self, time, outputs, state, sample_ids, name=None):
    #     """next_inputs_fn for GreedyEmbeddingHelper."""
    #     del outputs  # unused by next_inputs_fn

    #     next_time = time + 1
    #     finished = (next_time >= self._sequence_length)
    #     all_finished = math_ops.reduce_all(finished)

    #     next_inputs = control_flow_ops.cond(
    #         all_finished,
    #         # If we're finished, the next_inputs value doesn't matter
    #         lambda: self._start_inputs,
    #         lambda: self._embedding_fn(sample_ids))
    #     return (finished, next_inputs, state)

class Seq2SeqNetwork:
    def __init__(self, name, hparams, reuse, encoder_inputs, decoder_inputs, decoder_full_length, decoder_targets ):
        self.encoder_units = hparams.encoder_units
        self.decoder_units = hparams.decoder_units
        self.n_features = hparams.n_features
        self.num_layers = hparams.num_layers
        self.unit_type = hparams.unit_type
        self.name = name
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.decoder_full_length = decoder_full_length
        self.is_attention = hparams.is_attention
        self.forget_bias=hparams.forget_bias
        self.dropout=hparams.dropout
        self.num_residual_layers = hparams.num_residual_layers
        self.encoder_hidden_unit = hparams.encoder_units
        self.reuse= reuse
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token
        self.learning_rate = hparams.learning_rate
        self._build_model()

    def _build_model(self):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse, initializer=tf.glorot_normal_initializer()):
            self.scope = tf.compat.v1.get_variable_scope().name
            dropout_rate = self.dropout  
            self.embeddings = tf.Variable(tf.random.uniform(
                    [self.n_features, self.encoder_hidden_unit],
                    -1.0, 1.0), dtype=tf.float32)
            
            self.encoder_embeddings = tf.contrib.layers.fully_connected(
                self.encoder_inputs,
                self.encoder_hidden_unit,
                activation_fn=tf.nn.relu,
                scope="encoder_embeddings",
                reuse=tf.compat.v1.AUTO_REUSE)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

            self.decoder_targets_embeddings = tf.one_hot(
                self.decoder_targets,
                self.n_features,
                dtype=tf.float32)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            initializer = tf.contrib.layers.xavier_initializer()
            self.output_layer = tf.compat.v1.layers.Dense(
                self.n_features,
                activation=tf.sigmoid,
                use_bias=False,
                kernel_initializer=initializer,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                name="output_projection")

            with tf.variable_scope('encoder'):
                encoder_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(self.decoder_units, reuse=tf.AUTO_REUSE, forget_bias=self.forget_bias),
                        output_keep_prob=1 - dropout_rate)
                    for _ in range(self.num_layers)
                ]) if self.unit_type == 'lstm' else tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.GRUCell(self.decoder_units, reuse=tf.AUTO_REUSE),
                        output_keep_prob=1 - dropout_rate)
                    for _ in range(self.num_layers)
                ])

                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    sequence_length=None,
                    inputs=self.encoder_embeddings,
                    dtype=tf.float32,
                    swap_memory=True
                )

            with tf.variable_scope('decoder'):
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(self.decoder_units, reuse=tf.AUTO_REUSE, forget_bias=self.forget_bias),
                        output_keep_prob=1 - dropout_rate)
                    for _ in range(self.num_layers)
                ]) if self.unit_type == 'lstm' else tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.GRUCell(self.decoder_units, reuse=tf.AUTO_REUSE),
                        output_keep_prob=1 - dropout_rate)
                    for _ in range(self.num_layers)
                ])

                if self.is_attention:
                    attention_mechanism = LuongAttention(self.decoder_units, encoder_outputs)
                    decoder_cell = AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.decoder_units)
                    decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=tf.shape(self.decoder_full_length))
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_final_state)

                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=False)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=decoder_initial_state,
                    output_layer=self.output_layer
                )

                self.decoder_outputs, self.decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                output_time_major=False,
                                                                                                maximum_iterations=self.decoder_full_length[0])
                self.decoder_logits = self.decoder_outputs.rnn_output
                self.pi = tf.nn.softmax(self.decoder_logits)
                self.q = tf.compat.v1.layers.dense(self.decoder_logits, self.n_features, activation=None,
                                        reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")
                self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)
                self.decoder_prediction = self.decoder_outputs.sample_id

                num_output_units = self.n_features  
                with tf.variable_scope('logits_layer'):
                    self.logits = tf.layers.dense(self.decoder_logits, num_output_units, name='logits')
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.decoder_logits, logits=self.logits))


                decoder_cell_sample = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(self.decoder_units, reuse=tf.AUTO_REUSE, forget_bias=self.forget_bias),
                        output_keep_prob=1 - dropout_rate)
                    for _ in range(self.num_layers)
                ]) if self.unit_type == 'lstm' else tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.GRUCell(self.decoder_units, reuse=tf.AUTO_REUSE),
                        output_keep_prob=1 - dropout_rate)
                    for _ in range(self.num_layers)
                ])

                if self.is_attention:
                    attention_mechanism_sample = LuongAttention(self.decoder_units, encoder_outputs)
                    decoder_cell_sample = AttentionWrapper(decoder_cell_sample, attention_mechanism_sample, attention_layer_size=self.decoder_units)
                    decoder_initial_state_sample = (
                    decoder_cell.zero_state(tf.size(self.decoder_full_length),
                                            dtype=tf.float32).clone(
                        cell_state=encoder_final_state))

                # helper_sample = FixedSequenceLearningSampleEmbedingHelper(
                helper_sample = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    # sequence_length=self.decoder_full_length,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token,
                    seed=42,
                    softmax_temperature=1.0
                )
                decoder_sample = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell_sample,
                    helper=helper_sample,
                    initial_state=decoder_initial_state_sample,
                    output_layer=self.output_layer
                )
                
                self.sample_decoder_outputs, self.sample_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder_sample, 
                    output_time_major=False,
                    maximum_iterations=self.decoder_full_length[0]
                )
                self.sample_decoder_logits = self.sample_decoder_outputs.rnn_output
                self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
                self.sample_q = tf.compat.v1.layers.dense(self.sample_decoder_logits, self.n_features,
                                                activation=None, reuse=tf.compat.v1.AUTO_REUSE, name="qvalue_layer")

                self.sample_vf = tf.reduce_sum(self.sample_pi*self.sample_q, axis=-1)

                self.sample_decoder_prediction = self.sample_decoder_outputs.sample_id
                self.sample_decoder_embeddings = tf.one_hot(self.sample_decoder_prediction,
                                                            self.n_features,
                                                            dtype=tf.float32)

                self.sample_neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.sample_decoder_embeddings,
                                                                                logits=self.sample_decoder_logits)
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                    100000, 0.96, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)  # Adjust clip_norm as needed
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))

    def train(self, session, encoder_inputs, decoder_inputs, decoder_targets, decoder_full_length):
        feed_dict = {
            self.encoder_inputs: encoder_inputs,
            self.decoder_inputs: decoder_inputs,
            self.decoder_targets: decoder_targets,
            self.decoder_full_length: decoder_full_length
        }
        _, logits, loss = session.run([self.train_op, self.logits, self.loss], feed_dict=feed_dict)
        return loss

    def save(self, session, save_path):
        self.saver.save(session, save_path)

    def load(self, session, save_path):
        self.saver.restore(session, save_path)
        
    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)
class Seq2SeqPolicy(object):
    def __init__(self, obs_dim, encoder_units,
                 decoder_units, vocab_size, name="pi"):
        # Define TensorFlow HParams
        self.hparams = tf.contrib.training.HParams(
            encoder_units = 128,        # Positive integer representing number of units in encoder layers
            decoder_units = 128,        # Positive integer representing number of units in decoder layers
            n_features = vocab_size,              # Positive integer representing size of vocabulary
            num_layers = 3,                       # Positive integer specifying number of layers in encoder and decoder
            unit_type="lstm",                     # String indicating type of recurrent unit (e.g., "lstm", "gru", "rnn")
            is_attention=True,                    # Boolean indicating whether attention mechanism is used (True/False)
            forget_bias=0,                      # Float value typically close to 1.0, used to initialize LSTM forget gate bias
            dropout=0.35,                            # Float value between 0 and 1 indicating dropout rate
            num_residual_layers=1,                # Non-negative integer specifying number of residual connections
            start_token=0,                        # Integer index for start token in vocabulary
            end_token=2,                          # Integer index for end token in vocabulary
            is_bidencoder=False,                  # Boolean indicating whether bidirectional encoding is used (True/False)
            learning_rate=1e-4
        )

        self.batch_size = 50
        self.max_seq_length = 50
        self.num_epochs = 100
        self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_targets_ph_"+name)
        self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs_ph"+name)
        self.obs = tf.compat.v1.placeholder(shape=[None, None, obs_dim], dtype=tf.float32, name="obs_ph"+name)
        self.decoder_full_length = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="decoder_full_length"+name)
        self.name=name
        self.obs_dim = obs_dim
        # self.network = Seq2SeqNetwork(
        #     name=self.name,
        #     hparams=self.hparams, 
        #     reuse=tf.AUTO_REUSE,
        #     encoder_inputs=self.obs, 
        #     decoder_inputs=self.decoder_inputs, 
        #     decoder_targets=self.decoder_targets, 
        #     decoder_full_length=self.decoder_full_length,
        #     )
        self.network = Seq2SeqNetwork( hparams = self.hparams, reuse=tf.compat.v1.AUTO_REUSE,
                 encoder_inputs=self.obs,
                 decoder_inputs=self.decoder_inputs,
                 decoder_full_length=self.decoder_full_length,
                 decoder_targets=self.decoder_targets,name = name)

        self.vf = self.network.vf
    def tran(self, task_samples, batch_size=50):
        self.num_inner_grad_steps=4

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_inner_grad_steps):
                total_loss = 0.0
                batch_number = int(task_samples['observations'].shape[0] / batch_size)
                # shift_actions = np.column_stack(
                #     (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), task_samples['actions'][:, 0:-1]))
                shift_actions = np.roll(task_samples['actions'], shift=1, axis=1)
                shift_actions[:, 0] = 0
                shift_action_batchs = np.split(np.array(shift_actions), batch_number)
                observations_batchs = np.split(np.array(task_samples['observations']), batch_number)
                decoder_full_length = np.array([observations_batchs[0].shape[1]] * observations_batchs[0].shape[0], dtype=np.int32)
                actions_batchs = np.split(np.array(task_samples['actions']), batch_number)
                for observations, actions, shift_actions in zip(observations_batchs, actions_batchs, shift_action_batchs):                
                    batch_input_data = observations
                    batch_decoder_input_data = shift_actions
                    batch_targets = actions
                    batch_decoder_full_length = decoder_full_length
                    # print(batch_input_data.shape, batch_decoder_input_data.shape, batch_targets.shape, batch_decoder_full_length.shape)
                    loss = self.network.train(sess, batch_input_data, batch_decoder_input_data, batch_targets, batch_decoder_full_length)
                    total_loss += loss
            avg_loss = total_loss / ( self.num_inner_grad_steps)
            print("#######################")
            print("Average Loss = {:.15f}".format( avg_loss))
            print("#######################")
            # self.policy.save(sess, 'seq2seq_model.ckpt')
            # print("Model saved.")
        # with tf.Session() as sess:
        #     self.policy.load(sess, 'seq2seq_model.ckpt')
        #     new_input_data = np.random.randn(self.max_seq_length, self.obs_dim)
        #     new_decoder_input_data = np.random.randint(0, self.hparams.n_features, size=(num_samples, self.max_seq_length))
        #     new_decoder_full_length = np.random.randint(1, self.max_seq_length + 1, size=(num_samples,))
        #     logits = sess.run(
        #         self.policy.logits, 
        #         feed_dict={self.policy.encoder_inputs: new_input_data, 
        #                    self.policy.decoder_inputs: new_decoder_input_data, 
        #                    self.policy.decoder_full_length: new_decoder_full_length
        #                    })
        #     print("Inference result:")
        #     print(logits)

    def get_actions(self, observations):
        sess = tf.compat.v1.get_default_session()

        decoder_full_length = np.array( [observations.shape[1]] * observations.shape[0] , dtype=np.int32)

        actions, logits, v_value = sess.run([self.network.sample_decoder_prediction,
                                             self.network.sample_decoder_logits,
                                             self.network.sample_vf],
                                            feed_dict={self.obs: observations, self.decoder_full_length: decoder_full_length})

        return actions, logits, v_value
    
    def get_variables(self):
        return self.network.get_variables()

    def get_trainable_variables(self):
        return self.network.get_trainable_variables()


class MetaSeq2SeqPolicy():
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units,
                 vocab_size):

        self.meta_batch_size = meta_batch_size
        self.obs_dim = obs_dim
        self.action_dim = vocab_size

        # self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units, vocab_size, name='core_policy')


        self.meta_policies = []

        self.assign_old_eq_new_tasks = []

        for i in range(meta_batch_size):
            self.meta_policies.append(Seq2SeqPolicy(obs_dim, encoder_units, decoder_units,
                                                    vocab_size, name="task_"+str(i)+"_policy"))

            # self.assign_old_eq_new_tasks.append(
            #     U.function([], [], updates=[tf.compat.v1.assign(oldv, newv)
            #                                 for (oldv, newv) in
            #                                 zipsame(self.meta_policies[i].get_variables(), self.core_policy.get_variables())])
            #     )

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

    # def async_parameters(self):
    #     # async_parameters.
    #     for i in range(self.meta_batch_size):
    #         self.assign_old_eq_new_tasks[i]()

    @property
    def distribution(self):
        return self._dist

