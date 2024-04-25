import tensorflow as tf
import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
logging.root.setLevel(logging.INFO)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.contrib.seq2seq import AttentionWrapper, LuongAttention

class Seq2SeqPolicy:
    def __init__(self, hparams, encoder_inputs, decoder_inputs, decoder_targets, decoder_full_length, name='pi', reuse=tf.AUTO_REUSE):
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
        self._build_model()

    def _build_model(self):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse, initializer=tf.glorot_normal_initializer()):
            dropout_rate = self.dropout  
            self.embeddings = tf.Variable(tf.random.uniform(
                    [self.n_features, self.encoder_hidden_unit],
                    -1.0, 1.0), dtype=tf.float32)
            
            self.encoder_embeddings = tf.contrib.layers.fully_connected(
                self.encoder_inputs,
                self.encoder_hidden_unit,
                activation_fn=None,
                scope="encoder_embeddings",
                reuse=tf.compat.v1.AUTO_REUSE)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

            self.decoder_targets_embeddings = tf.one_hot(
                self.decoder_targets,
                self.n_features,
                dtype=tf.float32)

            self.output_layer = tf.compat.v1.layers.Dense(
                self.n_features,
                use_bias=False,
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

                encoder_inputs_reshaped = tf.transpose(self.encoder_inputs, [1, 0, 2])
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    sequence_length=None,
                    inputs=encoder_inputs_reshaped,
                    dtype=tf.float32,
                    time_major=True,
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
                    decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=tf.shape(self.encoder_inputs)[0])
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_final_state)

                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=True)

                decoder_inputs_reshaped = tf.transpose(self.decoder_inputs, [1, 0])
                decoder_full_length_reshaped = self.decoder_full_length

                self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(
                    cell=decoder_cell,
                    sequence_length=self.decoder_full_length,
                    inputs=self.decoder_embeddings,
                    dtype=tf.float32,
                    initial_state=decoder_initial_state,
                    time_major=True,
                    swap_memory=True
                )

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=decoder_initial_state,
                    output_layer=self.output_layer
                )

                self.decoder_outputs, self.decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                output_time_major=True,
                                                                                                maximum_iterations=self.decoder_full_length[0])
                self.decoder_logits = self.decoder_outputs.rnn_output

                # Ensure that the number of output units in the logits matches the dimensionality of the target data
                num_output_units = self.n_features  # Assuming self.n_features represents the number of output units
                with tf.variable_scope('logits_layer'):
                    self.logits = tf.layers.dense(self.decoder_logits, num_output_units, name='logits')
                self.loss = tf.reduce_mean(tf.square(self.logits - self.decoder_logits))

            # Ensure that the optimizer is minimizing the correct loss
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.loss)
            self.saver = tf.train.Saver()

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

class Seq2SeqPolicyNetwork(object):
    def __init__(self, obs_dim=17, name='pi'):
        self.hparams = tf.contrib.training.HParams(
            encoder_units = 128,
            decoder_units = 128,
            n_features = 2,
            num_layers = 2,
            unit_type="lstm",
            is_attention=True,
            forget_bias=1.0,
            dropout=0,
            num_residual_layers=0,
            start_token=0,
            end_token=2,
            is_bidencoder=False
        )
        self.batch_size = 50
        self.max_seq_length = 50
        self.num_epochs = 100
        self.encoder_inputs = self.obs = tf.compat.v1.placeholder(shape=[None, None, obs_dim], dtype=tf.float32, name="encoder_inputs")
        self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs_ph")
        self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_targets_ph")
        self.decoder_full_length = tf.compat.v1.placeholder(tf.int32, shape=(None,), name="decoder_full_length")

        self.obs_dim = obs_dim

    def tran(self):
        self.policy = Seq2SeqPolicy(
            self.hparams, 
            self.encoder_inputs, 
            self.decoder_inputs, 
            self.decoder_targets, 
            self.decoder_full_length,
            )
        num_samples = 1000
        train_input_data = np.random.randn(num_samples, self.max_seq_length, self.obs_dim)
        train_decoder_input_data = np.random.randint(0, self.hparams.n_features, size=(num_samples, self.max_seq_length))
        train_targets = np.random.randint(0, self.hparams.n_features, size=(num_samples, self.max_seq_length))
        train_decoder_full_length = np.random.randint(1, self.max_seq_length + 1, size=(num_samples,))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.num_epochs):
                total_loss = 0.0
                for i in range(0, num_samples, self.batch_size):
                    batch_input_data = train_input_data[i:i+self.batch_size]
                    batch_decoder_input_data = train_decoder_input_data[i:i+self.batch_size]
                    batch_targets = train_targets[i:i+self.batch_size]
                    batch_decoder_full_length = train_decoder_full_length[i:i+self.batch_size]
                    # print(batch_input_data.shape, batch_decoder_input_data.shape, batch_targets.shape, batch_decoder_full_length.shape)
                    loss = self.policy.train(sess, batch_input_data, batch_decoder_input_data, batch_targets, batch_decoder_full_length)
                    total_loss += loss
                avg_loss = total_loss / (num_samples // self.batch_size)
                print("Epoch {}: Average Loss = {:.15f}".format(epoch+1, avg_loss))
            self.policy.save(sess, 'seq2seq_model.ckpt')
            print("Model saved.")
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

        actions, logits, v_value = sess.run([self.policy.decoder_prediction,
                                             self.policy.decoder_logits,
                                             self.policy.sample_vf],
                                            feed_dict={self.obs: observations, self.decoder_full_length: decoder_full_length})

        return actions, logits, v_value

if __name__ == "__main__":
    check = Seq2SeqPolicyNetwork()
    check.tran()
