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
    def __init__(self, hparams, encoder_inputs, decoder_inputs, targets, encoder_seq_lengths, decoder_seq_lengths, name='pi'):
        self.encoder_units = hparams.encoder_units
        self.decoder_units = hparams.decoder_units
        self.n_features = hparams.n_features
        self.num_layers = hparams.num_layers
        self.unit_type = hparams.unit_type
        self.name = name
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.targets = targets
        self.encoder_seq_lengths = encoder_seq_lengths
        self.decoder_seq_lengths = decoder_seq_lengths
        self.is_attention = hparams.is_attention
        self.forget_bias=hparams.forget_bias
        self.dropout=hparams.dropout
        self.num_residual_layers = hparams.num_residual_layers

        self._build_model()

    def _build_model(self):
        # Dropout rate
        dropout_rate = self.dropout  

        # Encoder
        with tf.variable_scope('encoder'):
            if self.unit_type == 'lstm':
                encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.decoder_units, reuse=tf.AUTO_REUSE, forget_bias=self.forget_bias), output_keep_prob=1-dropout_rate) for _ in range(self.num_layers)])
            elif self.unit_type == 'gru':
                encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.decoder_units, reuse=tf.AUTO_REUSE), output_keep_prob=1-dropout_rate) for _ in range(self.num_layers)])
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, dtype=tf.float32, sequence_length=self.encoder_seq_lengths)

        # Decoder
        with tf.variable_scope('decoder'):
            if self.unit_type == 'lstm':
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.decoder_units, reuse=tf.AUTO_REUSE,forget_bias=self.forget_bias), output_keep_prob=1-dropout_rate) for _ in range(self.num_layers)])
            elif self.unit_type == 'gru':
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.decoder_units, reuse=tf.AUTO_REUSE), output_keep_prob=1-dropout_rate) for _ in range(self.num_layers)])

            if self.is_attention:
                attention_mechanism = LuongAttention(self.decoder_units, encoder_outputs, memory_sequence_length=self.encoder_seq_lengths)
                decoder_cell = AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.decoder_units)
                # Create an AttentionWrapperState as the initial state
                decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=tf.shape(self.encoder_inputs)[0])
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_final_state)

        decoder_outputs, _ = tf.nn.dynamic_rnn(decoder_cell, self.decoder_inputs, initial_state=decoder_initial_state, dtype=tf.float32, sequence_length=self.decoder_seq_lengths)
        # Output layer
        self.logits = tf.layers.dense(decoder_outputs, self.n_features, name='logits')

        # Define loss
        self.loss = tf.reduce_mean(tf.square(self.logits - self.targets))

        # Define optimizer
        self.optimizer = tf.train.AdamOptimizer()

        # Training operation
        self.train_op = self.optimizer.minimize(self.loss)

        # Saver
        self.saver = tf.train.Saver()


    def train(self, session, encoder_inputs, decoder_inputs, targets, encoder_seq_lengths, decoder_seq_lengths):
        feed_dict = {
            self.encoder_inputs: encoder_inputs,
            self.decoder_inputs: decoder_inputs,
            self.targets: targets,
            self.encoder_seq_lengths: encoder_seq_lengths,
            self.decoder_seq_lengths: decoder_seq_lengths
        }
        _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

class Seq2SeqPolicyNetwork(object):
    # Define hyperparameters
    def __init__(self):
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
        self.batch_size = 32
        self.max_seq_length = 50
        self.num_epochs = 100
        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, self.hparams.encoder_units], name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(tf.float32, [None, None, self.hparams.n_features], name='decoder_inputs')
        self.targets = tf.placeholder(tf.float32, [None, None, self.hparams.n_features], name='targets')
        self.encoder_seq_lengths = tf.placeholder(tf.int32, [None], name='encoder_seq_lengths')
        self.decoder_seq_lengths = tf.placeholder(tf.int32, [None], name='decoder_seq_lengths')    

    def tran(self):
        policy = Seq2SeqPolicy(
            self.hparams, 
            self.encoder_inputs, 
            self.decoder_inputs, 
            self.targets, 
            self.encoder_seq_lengths, 
            self.decoder_seq_lengths)
        num_samples = 1000
        train_input_data = np.random.randn(num_samples, self.max_seq_length, self.hparams.encoder_units)
        train_decoder_input_data = np.random.randn(num_samples, self.max_seq_length, self.hparams.n_features)
        train_targets = np.random.randn(num_samples, self.max_seq_length, self.hparams.n_features)
        train_encoder_seq_lengths = np.random.randint(1, self.max_seq_length + 1, size=num_samples)
        train_decoder_seq_lengths = np.random.randint(1, self.max_seq_length + 1, size=num_samples)

        # Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.num_epochs):
                total_loss = 0.0
                for i in range(0, num_samples, self.batch_size):
                    batch_input_data = train_input_data[i:i+self.batch_size]
                    batch_decoder_input_data = train_decoder_input_data[i:i+self.batch_size]
                    batch_targets = train_targets[i:i+self.batch_size]
                    batch_encoder_seq_lengths = train_encoder_seq_lengths[i:i+self.batch_size]
                    batch_decoder_seq_lengths = train_decoder_seq_lengths[i:i+self.batch_size]

                    loss = policy.train(sess, batch_input_data, batch_decoder_input_data, batch_targets, batch_encoder_seq_lengths, batch_decoder_seq_lengths)
                    total_loss += loss

                avg_loss = total_loss / (num_samples // self.batch_size)
                print("Epoch {}: Average Loss = {:.4f}".format(epoch+1, avg_loss))

            # Save the trained model
            policy.save(sess, 'seq2seq_model.ckpt')
            print("Model saved.")

        # Example of loading the saved model and performing inference
        with tf.Session() as sess:
            policy.load(sess, 'seq2seq_model.ckpt')

            # Generate some new input data for inference
            new_input_data = np.random.randn(1, self.max_seq_length, self.hparams.encoder_units)
            new_decoder_input_data = np.random.randn(1, self.max_seq_length, self.hparams.n_features)
            new_encoder_seq_lengths = np.array([self.max_seq_length])
            new_decoder_seq_lengths = np.array([self.max_seq_length])

            # Perform inference
            logits = sess.run(policy.logits, feed_dict={policy.encoder_inputs: new_input_data, policy.decoder_inputs: new_decoder_input_data, policy.encoder_seq_lengths: new_encoder_seq_lengths, policy.decoder_seq_lengths: new_decoder_seq_lengths})
            print("Inference result:")
            print(logits)

if __name__ == "__main__":
    check = Seq2SeqPolicyNetwork()
    check.tran()
