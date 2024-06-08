import tensorflow as tf
import numpy as np
import time
from utils import logger
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.contrib.distributions import Categorical
from policies.distributions.categorical_pd import CategoricalPd
import joblib
import os
import glob

class Trainer():
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                batch_size=500,
                start_itr=0,
                num_inner_grad_steps=3):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.batch_size = batch_size

    def train(self):
        """
        Implement the repilte algorithm for ppo reinforcement learning
        """
        start_time = time.time()
        avg_ret = []
        avg_pg_loss = []
        avg_vf_loss = []

        avg_latencies = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            # logger.log("\n ---------------- Iteration %d ----------------" % itr)
            # logger.log("Sampling set of tasks/goals for this meta-batch...")

            paths = self.sampler.obtain_samples(log=True, log_prefix='')

            """ ----------------- Processing Samples ---------------------"""
            # logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log='all', log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.batch_size)

            #print("task losses: ", losses)
            # print("average policy losses: ", np.mean(policy_losses))
            avg_pg_loss.append(np.mean(policy_losses))

            # print("average value losses: ", np.mean(value_losses))
            avg_vf_loss.append(np.mean(value_losses))

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)

            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.dumpkvs()
            avg_ret.append(avg_reward)

        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies
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

if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    # from policies.meta_seq2seq_policy import  Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO
    from utils import utils, logger

    logger.configure(dir="./meta_evaluate_ppo_log/task_offloading", format_strs=['stdout', 'log', 'csv'])

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_12/random.20."
                                    ],
                                time_major=False)

    print("calculate baseline solution======")

    env.set_task(0)
    action, finish_time = env.greedy_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(action[env.task_id],
                                          env.task_graphs_batchs[env.task_id],
                                          env.max_running_time_batchs[env.task_id],
                                          env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg greedy solution: ", np.mean(discounted_reward))
    print("avg greedy solution: ", np.mean(task_finish_time_batch))
    print("avg greedy solution: ", np.mean(finish_time))

    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))
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
    policy = Seq2SeqPolicy(obs_dim=17,
                           hparams=hparams, 
                           vocab_size=2,
                           name="core_policy")

    sampler = Seq2SeqSampler(env,
                             policy,
                             rollouts_per_meta_task=1,
                             max_path_length=20000,
                             envs_per_task=None,
                             parallel=False)

    baseline = ValueFunctionBaseline()

    sample_processor = Seq2SeSamplerProcessor(baseline=baseline,
                                              discount=0.99,
                                              gae_lambda=0.95,
                                              normalize_adv=True,
                                              positive_adv=False)
    algo = PPO(policy=policy,
               meta_sampler=sampler,
               meta_sampler_process=sample_processor,
               lr=1e-4,
               num_inner_grad_steps=3,
               clip_value=0.2,
               max_grad_norm=None)

    # define the trainer of ppo to evaluate the performance of the trained meta policy for new tasks.
    trainer = Trainer(algo=algo,
                      env=env,
                      sampler=sampler,
                      sample_processor=sample_processor,
                      policy=policy,
                      n_itr=2,
                      start_itr=0,
                      batch_size=500,
                      num_inner_grad_steps=3)

    # checkpoint_directory = "./meta_model_inner_step1"

    # Find all .ckpt files in the directory
    import re
    checkpoint_files = glob.glob("meta_model_inner_step1/meta_model_*.ckpt")

    def extract_number(filename):
        # Extract the number from the filename
        number = re.search(r'\d+', os.path.basename(filename))
        return int(number.group()) if number else 0
    checkpoint_files.sort(key=extract_number)
    # Loop through the found checkpoint files
    for checkpoint_file in checkpoint_files:
        with tf.compat.v1.Session() as sess:  # Use tf.compat.v1.Session for TensorFlow 1.x
            sess.run(tf.compat.v1.global_variables_initializer())

            # Load variables from the current checkpoint file
            load_path = checkpoint_file  
            policy.load_variables(load_path)
            print(checkpoint_file)
            # Train and evaluate on this checkpoint
            avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()

            # (Optionally) Log or store the results for this checkpoint
            # print(f"Results for checkpoint {checkpoint_file}:")
            # print(f"  Average Return: {avg_ret}")
            # print(f"  Average PG Loss: {avg_pg_loss}")
            # print(f"  Average VF Loss: {avg_vf_loss}")
            # print(f"  Average Latencies: {avg_latencies}")


