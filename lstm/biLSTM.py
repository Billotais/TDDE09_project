import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from lstm.generic_model import GenericModel
import logging

logging.getLogger().setLevel(logging.INFO)


class BiLSTM(GenericModel):
    def __init__(self, config, tag_dict_inv, sent_len, embeddings, n_tags, word_dict):
        super().__init__(config)
        self.tag_dict_inv = tag_dict_inv
        self.sent_len = sent_len
        self.word_dict = word_dict

        # Def placeholders for the computational graph
        self.word_ids = tf.placeholder(tf.int32, shape=[None, sent_len],
                                       name="word_ids")  # shape = (batch_size,sent_len)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")  # shape = (batch_size)
        self.labels = tf.placeholder(tf.int32, shape=[None, sent_len], name="labels")  # shape = (batch_size,sent_len)
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")

        # Def embeddings
        with tf.variable_scope("word_embeddings"):
            emb = tf.Variable(embeddings, dtype=tf.float32, trainable=False, name="_embeddings")
            embeddings = tf.nn.embedding_lookup(emb, self.word_ids, name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(embeddings, self.dropout)

        # Def Bi-LSTM
        with tf.variable_scope("encoder"):
            cell_fw = rnn.LSTMCell(self.config['embedding_dim'])
            cell_bw = rnn.LSTMCell(self.config['embedding_dim'])
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_bw, cell_fw=cell_fw,
                                                                  inputs=self.word_embeddings,
                                                                  sequence_length=self.seq_len, dtype=tf.float32)
            out = tf.concat([out_fw, out_bw], axis=-1)
            out = tf.nn.dropout(out, self.dropout)

        with tf.variable_scope("decoder"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2 * self.config['embedding_dim'], n_tags])
            b = tf.get_variable("b", dtype=tf.float32, shape=[n_tags], initializer=tf.zeros_initializer())
            n_steps = tf.shape(out)[1]
            out = tf.reshape(out, [-1, 2 * self.config['embedding_dim']])
            pred = tf.nn.xw_plus_b(out, W, b)
            self.logits = tf.reshape(pred, [-1, n_steps, n_tags], name="logits")
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32, name="labels_pred")

        with tf.variable_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.seq_len)
            xentropy = tf.boolean_mask(xentropy, mask)
            self.loss = tf.reduce_mean(xentropy, name="loss")
            tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("train"):
            clip = -1
            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(self.config['learning_rate'], global_step,
                                                       self.config['decay_steps'], self.config['decay_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            if clip > 0:
                gradients, vs = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, clip)
                self.training_op = optimizer.apply_gradients(zip(gradients, vs))
            else:
                self.training_op = optimizer.minimize(self.loss)

        # Initialize variables
        self.init_sess()

    def train(self, train_x, train_y, dev_x, dev_y, sequence_len, sequence_len_dev):
        """Training with early stopping"""
        best_acc = 0
        no_improvement = 0  # number of epochs without improvement for early stopping
        self.add_to_tensorboard()

        for epoch in range(self.config['n_epochs']):
            logging.info("Epoch {:} out of {:}".format(epoch + 1, self.config['n_epochs']))
            for i, (words, labels, seq_len_batch) in enumerate(
                    self.iterate_minibatches(train_x, train_y, sequence_len, self.config['batch_size'])):
                feed_dict = self.get_feed_dict(words, seq_len_batch, labels)
                _, train_loss, summary = self.sess.run([self.training_op, self.loss, self.merged], feed_dict=feed_dict)

            accuracy = self.evaluate(dev_x, dev_y, sequence_len_dev)

            if accuracy >= best_acc:
                no_improvement = 0
                self.save_sess()
                best_acc = accuracy
                logging.info("New best score: {:04.2f}%".format(accuracy))
            else:
                no_improvement += 1
                if no_improvement >= self.config['n_epoch_no_imp']:
                    logging.info("Early stopping: {} epochs without improvement".format(no_improvement))
                    break

    def get_feed_dict(self, words, sequence_len, labels=None):
        feed = {
            self.word_ids: words,
            # self.seq_len: sequence_len+2,
            self.seq_len: sequence_len,
            self.learning_rate: self.config['learning_rate'],
            self.dropout: self.config['dropout']
        }
        if labels is not None:
            feed[self.labels] = labels
        return feed

    def predict_batch(self, sents, sequence_len):
        feed_dict = self.get_feed_dict(sents, sequence_len)
        labels_pred = self.sess.run(self.labels_pred, feed_dict=feed_dict)

        return labels_pred

    def evaluate(self, test_sent, test_label, sequence_len):
        accs = []

        for sents, labels, seq_len_batch in self.iterate_minibatches(test_sent, test_label, sequence_len,
                                                                     self.config['batch_size']):
            pred = self.predict_batch(sents, seq_len_batch)
            for label_pred, label_gold, length in zip(pred, labels, seq_len_batch):
                label_pred = label_pred[:int(length)]
                label_gold = label_gold[:int(length)]
                accs += [pred == gold for (pred, gold) in zip(label_pred, label_gold)]
        acc = np.mean(accs) * 100
        return acc

    def tag(self, sentence):
        '''Returns list of tags'''
        sentence = [ self.word_dict[w] if w in self.word_dict  else 0 for w in sentence]
        sent_len = len(sentence)
        sentence = sentence + [self.word_dict['<PAD/>']]*(159-sent_len)
        sentence = np.reshape(sentence,(1,159))
        sent_len_r = np.reshape(sent_len, (1))
        pred_ids = self.predict_batch(sentence, sent_len_r)
        pred_tags = [self.tag_dict_inv[str(idx)] for idx in list(pred_ids[0])]
        pred_tags = pred_tags[:sent_len]
        return pred_tags
