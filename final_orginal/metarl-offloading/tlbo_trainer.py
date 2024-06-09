import os
import joblib

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions import categorical
# tf.get_logger().setLevel('WARNING')

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.contrib.distributions import Categorical
import utils as U
from utils.utils import zipsame
import itertools
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from env.mec_offloaing_envs.offloading_env import Resources
from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from policies.distributions.categorical_pd import CategoricalPd
import sys    
import logging
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


def _single_cell(hparams, residual_connection=False, residual_fn=None):
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


def create_rnn_cell(hparams, residual_fn=None, single_cell_fn=None):
    if not hparams.single_cell_fn:
        single_cell_fn = _single_cell

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
            encoder_cell = create_rnn_cell(hparams=hparams,)
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
            decoder_cell = create_rnn_cell(hparams=hparams)
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
        self.hparams =hparams
        self.build_network()
    
    def build_network(self):
        self.core_policy = Seq2SeqPolicy(self.obs_dim, self.action_dim, hparams=self.hparams, name='core_policy')
        self.meta_policies = []
        self.assign_old_eq_new_tasks = []
        for i in range(self.meta_batch_size):
            self.meta_policies.append(
                Seq2SeqPolicy(self.obs_dim, self.action_dim, hparams=self.hparams, name="task_" + str(i) + "_policy"))
            self.assign_old_eq_new_tasks.append(
                U.function([], [], updates=[tf.compat.v1.assign(oldv, newv)
                                            for (oldv, newv) in
                                            zipsame(self.meta_policies[i].get_variables(),
                                                    self.core_policy.get_variables())])
            )
        self._dist = CategoricalPd(self.action_dim)

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

    # @property
    # def distribution(self):
    #     return self._dist
# import logging
# import warnings
# import tensorflow as tf
# import numpy as np
# import time
# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger('tensorflow').disabled = True
# warnings.filterwarnings("ignore", category=UserWarning, module="gym")
# import tensorflow as tf
# import numpy as np
# import itertools

# this is the tf graph version of reptile:
import tensorflow as tf

class MRLCO():
    def __init__(self, policy, meta_batch_size, outer_lr, inner_lr, num_inner_grad_steps, clip_value=0.2, vf_coef=0.5, max_grad_norm=0.5):
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.policy = policy
        self.meta_batch_size = meta_batch_size
        self.update_numbers = 1
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.new_logits = [0.0] * self.meta_batch_size
        self.decoder_inputs = [0.0] * self.meta_batch_size
        self.old_logits = [0.0] * self.meta_batch_size
        self.actions = [0.0] * self.meta_batch_size
        self.obs = [0.0] * self.meta_batch_size
        self.vpred = [0.0] * self.meta_batch_size
        self.decoder_full_length = [0.0] * self.meta_batch_size

        self.old_v = [0.0] * self.meta_batch_size
        self.advs = [0.0] * self.meta_batch_size
        self.r = [0.0] * self.meta_batch_size        
        self.build_graph()

    def build_graph(self):
        self.surr_obj = [0.0] * self.meta_batch_size
        self.vf_loss = [0.0] * self.meta_batch_size
        self.total_loss = [0.0] * self.meta_batch_size
        self._train = [0.0] * self.meta_batch_size
        for i in range(int(self.meta_batch_size)):
            with tf.compat.v1.variable_scope("inner_update_parameters_task_"+str(i)) as scope:
                self.new_logits[i] = self.policy.meta_policies[i].network.decoder_logits
                self.decoder_inputs[i] = self.policy.meta_policies[i].decoder_inputs
                self.old_logits[i] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim], name='old_logits_ph_task_'+str(i))
                self.actions[i] = self.policy.meta_policies[i].decoder_targets
                self.obs[i] = self.policy.meta_policies[i].obs
                self.vpred[i] = self.policy.meta_policies[i].vf
                self.decoder_full_length[i] = self.policy.meta_policies[i].decoder_full_length

                self.old_v[i] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='old_v_ph_task_'+str(i))
                self.advs[i] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='advs_ph_task'+str(i))
                self.r[i] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='r_ph_task_'+str(i))                
                likelihood_ratio = self.policy._dist.likelihood_ratio_sym(self.actions[i], self.old_logits[i], self.new_logits[i])
                clipped_obj = tf.minimum(likelihood_ratio * self.advs[i], tf.clip_by_value(likelihood_ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * self.advs[i])
                self.surr_obj[i] = -tf.reduce_mean(clipped_obj)
                vpredclipped = self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i], -self.clip_value, self.clip_value)
                vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                vf_losses2 = tf.square(vpredclipped - self.r[i])
                self.vf_loss[i] = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                self.total_loss[i] = self.surr_obj[i] + self.vf_coef * self.vf_loss[i]
                params = self.policy.meta_policies[i].network.get_trainable_variables()
                grads_and_var = self.inner_optimizer.compute_gradients(self.total_loss[i], params)
                grads, var = zip(*grads_and_var)
                if self.max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_var = list(zip(grads, var))

                self._train[i] = self.inner_optimizer.apply_gradients(grads_and_var)

        with tf.compat.v1.variable_scope("outer_update_parameters") as scope:
            core_network_parameters = self.policy.core_policy.get_trainable_variables()
            self.grads_placeholders = [] 

            for i, var in enumerate(core_network_parameters):
                self.grads_placeholders.append(tf.compat.v1.placeholder(shape=var.shape, dtype=var.dtype, name="grads_"+str(i)))

            outer_grads_and_var = list(zip(self.grads_placeholders, core_network_parameters))
            self._outer_train = self.outer_optimizer.apply_gradients(outer_grads_and_var)


    def UpdateMetaPolicy(self, sess):
        # get the parameters value of the policy network
        # sess = tf.compat.v1.get_default_session()

        for i in range(int(self.meta_batch_size)):
            params_symbol = self.policy.meta_policies[i].get_trainable_variables()
            core_params_symble = self.policy.core_policy.get_trainable_variables()
            params = sess.run(params_symbol)
            core_params = sess.run(core_params_symble)

            update_feed_dict = {}

            # calcuate the gradient updates for the meta policy through first-order approximation.
            for i, core_var, meta_var in zip(itertools.count(), core_params, params):
                grads = (core_var - meta_var) / self.inner_lr / self.num_inner_grad_steps / self.meta_batch_size / self.update_numbers
                update_feed_dict[self.grads_placeholders[i]] = grads

            # update the meta policy parameters.
            _ = sess.run(self._outer_train, feed_dict=update_feed_dict)

        # print("async core policy to meta-policy")
        # self.policy.async_parameters()

    def UpdatePPOTarget(self, task_samples, sess, batch_size=50):
        total_policy_losses = []
        total_value_losses = []
        for i in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(task_samples[i], i, sess, batch_size)
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)

        return total_policy_losses, total_value_losses

    def UpdatePPOTargetPerTask(self, task_samples, task_id, sess, batch_size=50):
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        self.update_numbers = batch_number
        #:q!
        # print("update number is: ", self.update_numbers)
        #observations = task_samples['observations']

        shift_actions = np.column_stack(
                    (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), task_samples['actions'][:, 0:-1]))

        observations_batchs = np.split(np.array(task_samples['observations']), batch_number)
        actions_batchs = np.split(np.array(task_samples['actions']), batch_number)
        shift_action_batchs = np.split(np.array(shift_actions), batch_number)

        old_logits_batchs = np.split(np.array(task_samples["logits"], dtype=np.float32 ), batch_number)
        advs_batchs = np.split(np.array(task_samples['advantages'], dtype=np.float32), batch_number)
        oldvpred = np.split(np.array(task_samples['values'], dtype=np.float32), batch_number)
        returns = np.split(np.array(task_samples['returns'], dtype=np.float32), batch_number)

        # sess = tf.compat.v1.get_default_session()

        vf_loss = 0.0
        pg_loss = 0.0
        # copy_policy.set_weights(self.policy.get_weights())
        for i in range(int(self.num_inner_grad_steps)):
            # action, old_logits, _ = copy_policy(observations)
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                                                                                        shift_action_batchs, advs_batchs, returns):
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                feed_dict = {self.old_logits[task_id]: old_logits, self.old_v[task_id]: old_v, self.obs[task_id]: observations, self.actions[task_id]: actions,
                            self.decoder_inputs[task_id]: shift_actions,
                             self.decoder_full_length[task_id]: decoder_full_length, self.advs[task_id]: advs, self.r[task_id]: r}

                _, value_loss, policy_loss = sess.run([self._train[task_id], self.vf_loss[task_id], self.surr_obj[task_id]], feed_dict=feed_dict)

                vf_loss += value_loss
                pg_loss += policy_loss

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses


class Trainer(object):
    def __init__(self, 
                 algo,
                 sampler,
                 sampler_processor,
                 policy,
                 n_itr,
                 sess,
                 greedy_finish_time,
                 start_itr=0,
                 inner_batch_size=500,
                 ):
        self.algo = algo
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.policy_losses = 100000
        self.avg_latency = 100000
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.sess = sess
        self.greedy_finish_time = greedy_finish_time
    def train(self):
        avg_loss = []
        for itr in range(self.start_itr, self.n_itr):
            task_ids = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='') 
            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')            
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size, sess=self.sess)
            avg_loss.append(np.mean(policy_losses))
            self.algo.UpdateMetaPolicy(sess=self.sess)
            latency = np.array([])
            for i in range(len(samples_data)):
                latency = np.concatenate((latency, samples_data[i]['finish_time']), axis=-1)
            avg_latency = np.mean(latency)
            greedy_latency = np.mean(greedy_run_time)
            diff_latency = greedy_latency -  avg_latency
            print(f"avg_latency == {avg_latency} -- greedy == {np.mean(greedy_run_time)} -- diff == {diff_latency} -- taskid == {task_ids}")            
        return avg_loss

class TLBO:
    def __init__(self, population_size, dim, bounds, iterations, trainer):
        self.population_size = population_size
        self.dim = dim
        self.bounds = bounds
        self.iterations = iterations
        self.trainer = trainer
        self.population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, dim))
        self.fitness = np.zeros(population_size)
        self.teacher = 1000000

    def evaluate_population(self, sess, weight_loss=0.5, weight_reward=0.5):
        for i in range(self.population_size):
            # inner_lr, outer_lr, num_units, encoder_units, decoder_hidden_unit, dropout, forget_bias, num_layers = self.population[i]
            inner_lr, outer_lr, num_inner_grad_steps, inner_batch_size = self.population[i]


            # self.trainer.policy.hparams.num_units = int(num_units)
            # self.trainer.policy.hparams.encoder_units = int(encoder_units)
            # self.trainer.policy.hparams.decoder_hidden_unit = int(encoder_units)
            # self.trainer.policy.hparams.dropout = dropout
            # self.trainer.policy.hparams.forget_bias = forget_bias
            # self.trainer.policy.hparams.num_layers = int(num_layers)

            # Set trainer hyperparameters:
            self.trainer.algo.inner_lr = inner_lr
            self.trainer.algo.outer_lr = outer_lr
            self.trainer.algo.num_inner_grad_steps = num_inner_grad_steps
            self.trainer.algo.inner_batch_size = inner_batch_size
            # self.trainer.policy.build_network()
            # self.trainer.algo.build_graph()


            # Train the model and get average loss
            avg_loss = np.mean(self.trainer.train()) 
            paths = self.trainer.sampler.obtain_samples(log=False, log_prefix='')            
            samples_data = self.trainer.sampler_processor.process_samples(paths, log=False, log_prefix='') 
            avg_reward = np.mean([np.sum(path["rewards"]) for path in samples_data])

            self.fitness[i] = -(-weight_loss * avg_loss + weight_reward * avg_reward)  # Combine loss and reward
            # print(i, np.argmin(self.fitness), self.fitness[i], inner_lr, outer_lr, num_units, dropout, forget_bias, num_layers)
            # print(i, np.argmin(self.fitness), self.fitness[i], inner_lr, outer_lr, num_inner_grad_steps, inner_batch_size, self.teacher)



    def teacher_phase(self):
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index]
        mean_solution = np.mean(self.population, axis=0)
        new_solution_list = [0.0] * self.population_size
        for i in range(self.population_size):
            tf = np.random.randint(1, 3)
            new_solution = self.population[i] + np.random.rand(self.dim) * (best_solution - tf * mean_solution)
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            new_solution_list[i]=new_solution
        tmp_solution = self.population.copy()
        tmp_fitness = self.fitness.copy()
        self.population=new_solution_list.copy()
        self.evaluate_population(sess=sess)
        for i in range(self.population_size):
            if tmp_fitness[i] < self.fitness[i]:
                self.fitness[i] = tmp_fitness[i]
                self.population[i] = tmp_solution[i]
                # print(f"index {i} rejected  {tmp_fitness[i]} -- {tmp_solution[i]} -- {self.fitness[i]}")
    def learner_phase(self):
        new_solution_list = [0.0] * self.population_size
        for i in range(self.population_size):
            j = i
            while j == i:
                j = np.random.randint(self.population_size)

            if self.fitness[i] < self.fitness[j]:
                new_solution_list[i] = self.population[i] + np.random.rand(self.dim) * (self.population[i] - self.population[j])
            else:
                new_solution_list[i] = self.population[i] + np.random.rand(self.dim) * (self.population[j] - self.population[i])

            new_solution_list[i] = np.clip(new_solution_list[i], self.bounds[:, 0], self.bounds[:, 1])
        tmp_solution=self.population.copy()
        tmp_fitness = self.fitness.copy()
        self.population=new_solution_list.copy()
        self.evaluate_population(sess=sess)
        for i in range(self.population_size):
            if tmp_fitness[i] < self.fitness[i]:
                self.fitness[i] = tmp_fitness[i]
                self.population[i] = tmp_solution[i]
                # print(f"index {i} rejected  {tmp_fitness[i]} -- {tmp_solution[i]} -- {self.fitness[i]}")

    def optimize(self, sess):
        self.evaluate_population(sess=sess)
        for _ in range(self.iterations):
            print(f"\n\n{_}\n\n")
            self.teacher_phase()
            self.learner_phase()
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.teacher :
                self.teacher = self.fitness[best_index]
                print(f"\n\n New_teacher == {self.teacher}\n\n")
            self.trainer.policy.async_parameters()
            self.trainer.policy.core_policy.save_variables(
                    save_path="./meta_model_inner_step1/meta_model_" + str(_) + ".ckpt")
        self.trainer.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index]
        return best_solution


if __name__ == "__main__":
    META_BATCH_SIZE = 2

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
                                ],
                                time_major=False)
    action, greedy_finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(greedy_finish_time))
    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))
    print()
    hparams = tf.contrib.training.HParams(
        unit_type="lstm",
        num_units=128,
        encoder_units=128,
        decoder_hidden_unit=128,
        n_features=2,
        is_attention=True,
        forget_bias=1.0,
        dropout=0,
        num_gpus=1,
        num_layers=2,
        num_residual_layers=0,
        start_token=0,
        end_token=2,
        reuse=tf.compat.v1.AUTO_REUSE,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        single_cell_fn=None,
        residual_fn=None,
    )
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, vocab_size=2, hparams=hparams)
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,  # This batch_size is confusing
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=False,
    )    
    baseline = ValueFunctionBaseline()
    sampler_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True,
                                                   positive_adv=False)    
    algo = MRLCO(policy=meta_policy,
                #  meta_sampler=sampler,
                #  meta_sampler_process=sample_processor,
                 inner_lr=5e-4,
                 outer_lr=5e-4,
                 meta_batch_size=META_BATCH_SIZE,
                 num_inner_grad_steps=1,
                 clip_value=0.3)
    with tf.compat.v1.Session() as sess:

        trainer = Trainer(algo=algo,
                        sampler=sampler,
                        sampler_processor=sampler_processor,
                        policy=meta_policy,
                        n_itr=1,
                        start_itr=0,
                        inner_batch_size=1000, 
                        sess=sess,
                        greedy_finish_time=greedy_finish_time)

        bounds = np.array([
            [1e-20, 5e-4],     # inner_lr range
            [1e-20, 5e-4],     # outer_lr range
            # [64, 256],        # num_units range 
            # [64, 256],        # encoder_units range 
            # [64, 256],        # decoder_hidden_unit range 
            # [0.0, 0.5],       # dropout range
            # [0.5, 2.0],       # forget_bias range 
            # [1, 5]           # num_layers range
            [10, 30], # num_inner_grad_steps
            [10, 30], # inner_batch_size
            # [10, 1000], # num_inner_grad_steps
            # [10, 2000], # inner_batch_size
        ])      
        tlbo = TLBO(population_size=15, dim=4, bounds=bounds, iterations=100, trainer=trainer)
        sess.run(tf.global_variables_initializer())
        inner_lr, outer_lr, num_inner_grad_steps, inner_batch_size = tlbo.optimize(sess)
        print(f"inner_lr = {inner_lr} ,outer_lr = {outer_lr}, num_inner_grad_steps = {num_inner_grad_steps}, inner_batch_size = {inner_batch_size}")
