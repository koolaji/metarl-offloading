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

class Seq2SeqPolicy:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, name='core_policy'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.name = name
        self._build_model()

    def _build_model(self):
        # Define placeholders
        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, self.input_dim], name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(tf.float32, [None, None, self.output_dim], name='decoder_inputs')
        self.targets = tf.placeholder(tf.float32, [None, None, self.output_dim], name='targets')
        self.encoder_seq_lengths = tf.placeholder(tf.int32, [None], name='encoder_seq_lengths')
        self.decoder_seq_lengths = tf.placeholder(tf.int32, [None], name='decoder_seq_lengths')

        # Encoder
        with tf.variable_scope('encoder'):
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_dim, reuse=tf.AUTO_REUSE) for _ in range(self.num_layers)])
            _, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, dtype=tf.float32, sequence_length=self.encoder_seq_lengths)

        # Decoder
        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_dim, reuse=tf.AUTO_REUSE) for _ in range(self.num_layers)])
            decoder_outputs, _ = tf.nn.dynamic_rnn(decoder_cell, self.decoder_inputs, initial_state=encoder_final_state, dtype=tf.float32, sequence_length=self.decoder_seq_lengths)

        # Output layer
        self.logits = tf.layers.dense(decoder_outputs, self.output_dim, name='logits')

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

    def save(self, session, save_path):
        self.saver.save(session, save_path)

    def load(self, session, save_path):
        self.saver.restore(session, save_path)

def main():
    # Define hyperparameters
    input_dim = 128
    hidden_dim = 128
    output_dim = 2
    num_layers = 2
    batch_size = 32
    max_seq_length = 50
    num_epochs = 100

    # Create an instance of Seq2SeqPolicy
    policy = Seq2SeqPolicy(input_dim, hidden_dim, output_dim, num_layers)

    # Generate some dummy data for training
    num_samples = 1000
    train_input_data = np.random.randn(num_samples, max_seq_length, input_dim)
    train_decoder_input_data = np.random.randn(num_samples, max_seq_length, output_dim)
    train_targets = np.random.randn(num_samples, max_seq_length, output_dim)
    train_encoder_seq_lengths = np.random.randint(1, max_seq_length + 1, size=num_samples)
    train_decoder_seq_lengths = np.random.randint(1, max_seq_length + 1, size=num_samples)

    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            total_loss = 0.0
            for i in range(0, num_samples, batch_size):
                batch_input_data = train_input_data[i:i+batch_size]
                batch_decoder_input_data = train_decoder_input_data[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                batch_encoder_seq_lengths = train_encoder_seq_lengths[i:i+batch_size]
                batch_decoder_seq_lengths = train_decoder_seq_lengths[i:i+batch_size]

                loss = policy.train(sess, batch_input_data, batch_decoder_input_data, batch_targets, batch_encoder_seq_lengths, batch_decoder_seq_lengths)
                total_loss += loss

            avg_loss = total_loss / (num_samples // batch_size)
            print("Epoch {}: Average Loss = {:.4f}".format(epoch+1, avg_loss))

        # Save the trained model
        policy.save(sess, 'seq2seq_model.ckpt')
        print("Model saved.")

    # Example of loading the saved model and performing inference
    with tf.Session() as sess:
        policy.load(sess, 'seq2seq_model.ckpt')

        # Generate some new input data for inference
        new_input_data = np.random.randn(1, max_seq_length, input_dim)
        new_decoder_input_data = np.random.randn(1, max_seq_length, output_dim)
        new_encoder_seq_lengths = np.array([max_seq_length])
        new_decoder_seq_lengths = np.array([max_seq_length])

        # Perform inference
        logits = sess.run(policy.logits, feed_dict={policy.encoder_inputs: new_input_data, policy.decoder_inputs: new_decoder_input_data, policy.encoder_seq_lengths: new_encoder_seq_lengths, policy.decoder_seq_lengths: new_decoder_seq_lengths})
        print("Inference result:")
        print(logits)

if __name__ == "__main__":
    main()
