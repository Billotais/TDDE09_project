import tensorflow as tf
import numpy as np
import logging
from neural_network.generic_model import GenericModel


logging.getLogger().setLevel(logging.INFO)



class NeuralNetwork(GenericModel):
    def __init__(self, config, embeddings_words, embeddings_tags, n_pred, feature_type):
        super().__init__(config)

        if feature_type == 'tagger_1':
            num_word_features = 5
            num_feat = 7
        else:
            num_word_features = 14
            num_feat = 28


        #Def placeholders for the computational graph
        self.word_ids = tf.placeholder(tf.int32, shape=[None, num_feat], name="word_ids") #shape = (batch_size,sent_len)
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels") #shape = (batch_size,sent_len)
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        # self.pred_scores = tf.placeholder(tf.float32, shape=[], name="pred_scores")

        

        #Def embeddings
        with tf.variable_scope("word_embeddings"):
            emb_words = tf.Variable(embeddings_words, dtype=tf.float32, trainable=False, name="_embeddings_w")
            emb_tags = tf.Variable(embeddings_tags, dtype=tf.float32, trainable=False, name="_embeddings_t")

            # Get embeddings for words
            embeddings_words = tf.nn.embedding_lookup(emb_words, self.word_ids[:,0:num_word_features], name="word_embeddings")

            # get embedding for tags. 
            embeddings_tags = tf.nn.embedding_lookup(emb_tags, self.word_ids[:,num_word_features:], name="tag_embeddings")

            # Conver to batch_size num_feat*embedding_dimension
            embeddings_words = tf.reshape(embeddings_words, [-1, embeddings_words.shape[1]*embeddings_words.shape[2]])
            embeddings_tags = tf.reshape(embeddings_tags, [-1, embeddings_tags.shape[1]*embeddings_tags.shape[2]])

            # Concatenate the features
            self.embeddings = tf.concat([embeddings_words, embeddings_tags], -1)

        self.layers = []

        num_hidden = self.config['num_hidden']

        #Def Neural net
        with tf.variable_scope("neural_net"):
            l2_reg = self.config['l2_reg']
            self.layers.append(tf.layers.dense(self.embeddings, num_hidden[0], kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), activation=tf.nn.relu))

            for i in range(1,len(num_hidden)):
                self.layers.append(tf.layers.dense(self.layers[i-1], num_hidden[i], kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), activation=tf.nn.relu))

            # Output layer
            self.layers.append(tf.layers.dense(self.layers[-1], n_pred, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), activation=None))

            self.pred_scores = self.layers[-1]
            
            self.labels_pred = tf.cast(tf.argmax(self.layers[-1], axis=-1), tf.int32, name="labels_pred")

        with tf.variable_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layers[-1], labels=self.labels)
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



        #Initialize variables
        self.init_sess()


    def train(self, train_x, train_y, dev_x, dev_y):
        """Training with early stopping"""
        best_acc = 0
        no_improvement = 0 # number of epochs without improvement for early stopping
        self.add_to_tensorboard()

        for epoch in range(self.config['n_epochs']):
            logging.info("Epoch {:} out of {:}".format(epoch + 1, self.config['n_epochs']))
            for i, (words, labels) in enumerate(self.iterate_minibatches(train_x, train_y, None, self.config['batch_size'])):
                feed_dict = self.get_feed_dict(words, labels)
                _, train_loss, summary = self.sess.run([self.training_op, self.loss, self.merged], feed_dict=feed_dict)

            logging.info("Training for epoch {} finished".format(epoch + 1))
            accuracy = self.evaluate(dev_x, dev_y)
            msg = "Epoch {} - Accuracy: {:04.2f}".format(epoch, accuracy)
            logging.info(msg)

            if accuracy >= best_acc:
                no_improvement = 0
                self.save_sess()
                best_acc = accuracy
                logging.info("New best score!")
            else:
                no_improvement += 1
                if no_improvement >= self.config['n_epoch_no_imp']:
                    logging.info("Early stopping: {} epochs without improvement".format(no_improvement))
                    break

    def get_feed_dict(self, words, labels=None):
        feed = {
            self.word_ids: words,
            self.learning_rate: self.config['learning_rate'],
            self.dropout: self.config['dropout']
        }
        if labels is not None:
            feed[self.labels] = labels
        return feed

    def evaluate(self, test_sent, test_label):
        accs = []

        for sents, labels in self.iterate_minibatches(test_sent, test_label, None, self.config['batch_size']):
            pred = self.predict_batch(sents)
            accs += [pred==labels]
        acc = np.mean(accs)*100
        return acc

    def predict(self, sentence, valid_pred=None):
        '''Returns list of tags'''
        feed_dict = self.get_feed_dict(sentence)
        labels_scores = self.sess.run(self.pred_scores, feed_dict=feed_dict)

        if valid_pred is not None:
            # set to inf here
            labels_scores = labels_scores[0]
            labels_scores[~valid_pred] = -100000

        pred_ids = np.argmax(labels_scores)
        
        #pred_tags = [self.tag_dict_inv[idx] for idx in list(pred_ids[0])]
        return pred_ids


    def predict_batch(self, sents):
        feed_dict = self.get_feed_dict(sents)
        labels_pred = self.sess.run(self.labels_pred, feed_dict=feed_dict)
        return labels_pred
    





























