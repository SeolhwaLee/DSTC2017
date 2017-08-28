import tensorflow as tf
import time
import os
import math
from data_process import minibatches
from general_utils import Progbar, print_sentence
import numpy as np


class MLP():
    def __init__ (self, input_size, num_neuron, num_classes, utter_embed, config):
        self.input_size = input_size
        self.num_neuron = num_neuron
        self.num_classes = num_classes
        self.utter_embed = utter_embed
        self.logger = config.logger
        self.config = config

        self.cate_mapping_dict = {'O':1, 'X':2, 'T':3}



    def add_placeholders(self):
        with tf.variable_scope('input') as scope:
            self.input_features = tf.placeholder(tf.float32, [1, 10, self.input_size], name='input_features')

    def add_logits_op(self):
        with tf.variable_scope('hidden_layer') as scope:
            # layer1
            weight = tf.get_variable('weight', [self.input_size, self.num_neuron], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.input_size)))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron, ]))
            # y1 = tf.matmul(x, W1) + b1
            z = tf.add(tf.matmul(self.input_features, weight), bias)
            y = tf.nn.relu(z)

            name = ['hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6', 'hidden_layer7',
                    'hidden_layer8', 'hidden_layer9']

            # {print(layer) for layer in sorted(set(name))}
            for i in range(1):
                with tf.variable_scope(name[i]) as scope:
                   # layer1
                   weight = tf.get_variable('weight', [self.input_size, self.num_neuron], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.num_neuron)))
                   bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron, ]))
                   # y1 = tf.matmul(x, W1) + b1
                   y = tf.nn.relu(tf.add(tf.matmul(y, weight), bias))

        with tf.variable_scope('output_layer') as scope:
            # layer2
            weight = tf.get_variable('weight',[self.num_neuron, self.num_classes], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.num_neuron)))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_classes, ]))
            y_output = tf.nn.softmax(tf.matmul(y, weight) + bias)


        with tf.variable_scope('loss') as scope:
            # cross-entropy 모델을 설정한다.
            self.y = y_output
            # y = tf.Print(y, [y])
            self.ground_label = tf.placeholder(tf.float32, [None, self.num_classes])
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ground_label * tf.log(y), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

            tf.summary.scalar('loss', self.train_step)

    def build(self):
        self.add_placeholders()
        self.add_logits_op()
        self.train()


    def run_epoch(self, sess, train_data, dev_data, epoch):
        """
        :param train_data: contains concatenated sentence(user and system list type) and ground_labels(O, T, X)
        :return: accuracy and f1 scroe
        """
        num_batches = (len(train_data) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=num_batches)

        for i, (concat_utter_list, ground_label) in enumerate(minibatches(train_data + dev_data + self.test_data[:300], self.config.batch_size)):
            input_features = []
            for each_utter_list in concat_utter_list:
                user_sentence = each_utter_list[0]
                system_sentence = each_utter_list[1]
                if self.config.embed_method == 'word2vec':
                    user_embedding = self.utter_embed.embed_utterance(user_sentence)
                    system_embedding = self.utter_embed.embed_utterance(system_sentence)
                    input_feature = np.concatenate((user_embedding, system_embedding), axis=0)
                    input_features.append(input_feature)

                if self.config.embed_method == 'word2vec':
                    input_features = np.array([input_features])

                input_features.append(input_feature)

            input_features = np.array([input_features])

            ground_label_list = []
            for label in ground_label:
                ground_label_list.append(self.cate_mapping_dict[label.strip().encode('utf-8  ')])
            ground_label_list = np.array([ground_label_list])

            self.feed_dict = {
                self.input_features: input_features,
                self.ground_label: ground_label_list
            }






    def train(self, train_data, dev_data):
        with tf.Session as sess:
            # variables need to be initialized before we can use them
            sess.run(tf.global_variables_initializer())

            # merge all summaries into a single "operation" which we can execute in a session
            merged = tf.summary.merge_all()

            timestamp = str(int(time.time()))
            writer = tf.summary.FileWriter(os.path.join("./", "model_summaries", timestamp), sess.graph)

            for step in range(1000):
                # batch_xs, batch_ys = mnist.train.next_batch(100)
                train_loss, summary = sess.run([self.train_step, merged], feed_dict=self.feed_dict)

                if step % 10 == 0:
                    writer.add_summary(summary, step)


            # 학습된 모델이 얼마나 정확한지를 출력한다.
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.ground_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={self.input_features: self.input_features, self.ground_label: self.test_set[300:]}))

