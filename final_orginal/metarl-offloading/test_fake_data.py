import tensorflow as tf
import numpy as np

tf.disable_v2_behavior()  

class LSTMEncoderDecoder:
    def __init__(self, hparams):
        self.hparams = hparams

        with tf.variable_scope("encoder_decoder"):
            # Embedding (using random uniform initialization)
            self.embeddings = tf.Variable(
                tf.random.uniform([self.hparams.vocab_size, self.hparams.embedding_dim], -1.0, 1.0),
                dtype=tf.float32
            )

            # Encoder LSTM
            self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hparams.hidden_units)

            # Decoder LSTM
            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hparams.hidden_units)

            # Output Layer
            self.output_layer = tf.layers.Dense(self.hparams.vocab_size)

    def __call__(self, encoder_input, decoder_input, initial_state=None):
        with tf.variable_scope("encoder_decoder", reuse=tf.AUTO_REUSE):
            # Encoder
            encoder_embedded = tf.nn.embedding_lookup(self.embeddings, encoder_input)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                self.encoder_cell, encoder_embedded, initial_state=initial_state, dtype=tf.float32
            )

            # Decoder
            decoder_embedded = tf.nn.embedding_lookup(self.embeddings, decoder_input)
            decoder_outputs, _ = tf.nn.dynamic_rnn(
                self.decoder_cell, decoder_embedded, initial_state=encoder_state, dtype=tf.float32
            )

            # Output
            output = self.output_layer(decoder_outputs)
        return output, encoder_state


# Hyperparameters
hparams = tf.contrib.training.HParams( # Use the corrected path for HParams
    vocab_size=10, 
    embedding_dim=8,
    hidden_units=16,
    batch_size=4,
    max_input_length=5,
    max_output_length=6,
    learning_rate=0.001,
)

# Toy Example Data
encoder_input_data = np.random.randint(hparams.vocab_size, size=(hparams.batch_size, hparams.max_input_length))
decoder_input_data = np.random.randint(hparams.vocab_size, size=(hparams.batch_size, hparams.max_output_length))
decoder_target_data = np.random.randint(hparams.vocab_size, size=(hparams.batch_size, hparams.max_output_length))  

# Model and Training Setup
model = LSTMEncoderDecoder(hparams)
optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
loss_fn = tf.losses.sparse_softmax_cross_entropy

# Build the model graph
dummy_encoder_input = tf.placeholder(tf.int32, [hparams.batch_size, hparams.max_input_length])
dummy_decoder_input = tf.placeholder(tf.int32, [hparams.batch_size, hparams.max_output_length])
_ = model(dummy_encoder_input, dummy_decoder_input)

# Operations for training
train_op = optimizer.minimize(loss_fn(decoder_target_data, model(encoder_input_data, decoder_input_data, None)[0]))
loss_op = loss_fn(decoder_target_data, model(encoder_input_data, decoder_input_data, None)[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(3):
        _, loss_value = sess.run([train_op, loss_op])
        print(f"Epoch {epoch + 1}, Loss: {loss_value}")

# Inference (Single Sequence Example)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    input_seq = encoder_input_data[0]
    initial_state = sess.run(model.encoder_cell.zero_state(1, dtype=tf.float32))
    output_seq = []

    for t in range(hparams.max_output_length):
        logits, initial_state = model(np.zeros([1, 1], dtype=np.int32),  
                                     np.reshape(np.array([output_seq[t - 1]]), [1, 1]) if t > 0 else np.zeros((1, 1), dtype=np.int32), 
                                     initial_state)
        predicted_token = np.argmax(logits.eval(), axis=-1)
        output_seq.append(predicted_token[0, 0]) 

    print("Input Sequence:", input_seq)
    print("Output Sequence:", output_seq)

