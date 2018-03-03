import os
import logging
import tensorflow as tf
import numpy as np

"""Generic class that implements tensorflow methods that are not specific to the LSTM model"""

class GenericModel(object):
    def __init__(self, config):
        self.config = config
        logging.getLogger().setLevel(logging.INFO)
        tf.reset_default_graph()
        self.sess = None
        self.saver = None

    def init_weight(self, scope_name):
        '''Inits weights of a given layer'''
        var = tf.contrib.framework.get_variables(scope = scope_name)
        init = tf.variables_initializer(var)
        self.sess.run(init)

    def init_sess(self):
        """Initializes the sess and its variables"""
        logging.info("Initializing session and saver")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_sess(self):
        """Loads model from given directory dir"""
        logging.info("Restore session from trained model: Weights loaded!")
        self.saver.restore(self.sess, self.config['model_dir'])

    def save_sess(self):
        """Saves a session and its weights"""
        if not os.path.exists(self.config['model_dir']):
            os.makedirs(self.config['model_dir'])
        self.saver.save(self.sess, self.config['model_dir'])

    def close_sess(self):
        """Closes a session"""
        self.sess.close()

    def add_to_tensorboard(self):
        """Prepares variables for the tensorboard"""
        self.merged = tf.summary.merge_all()
        self.f_writer = tf.summary.FileWriter(self.config['model_dir'], self.sess.graph)

    def iterate_minibatches(self, inputs, targets, seq_len, batchsize, shuffle=True):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt], seq_len[excerpt]















































