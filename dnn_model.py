import tensorflow as tf
import time
import os
import math
from data_process import minibatches
from general_utils import Progbar, print_sentence
import numpy as np


class dnn():
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
            self.ground_label = tf.placeholder(tf.int32, [1, 10], name='ground_label')

    def add_logits_op(self):
        with tf.variable_scope('hidden_layer') as scope:
            reshaped_features = tf.transpose(self.input_features, [1, 0, 2])
            print('reshaped_features: ', reshaped_features.shape)
            reshaped_features = tf.reshape(reshaped_features, [-1, self.input_size])

            # layer1
            weight = tf.get_variable('weight', [self.input_size, self.num_neuron], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.input_size)))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron, ]))
            # y1 = tf.matmul(x, W1) + b1
            z = tf.add(tf.matmul(reshaped_features, weight), bias)
            y = tf.nn.relu(z)
            print("y shape:", y.shape)

            name = ['hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6', 'hidden_layer7',
                    'hidden_layer8', 'hidden_layer9']

            # {print(layer) for layer in sorted(set(name))}
            # for i in range(1):
            #     with tf.variable_scope(name[i]) as scope:
            #        # layer1
            #        weight = tf.get_variable('weight', [self.input_size, self.num_neuron], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.num_neuron)))
            #        bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron, ]))
            #        # y1 = tf.matmul(x, W1) + b1
            #        y = tf.nn.relu(tf.add(tf.matmul(y, weight), bias))

        with tf.variable_scope('output_layer') as scope:
            # layer2
            weight = tf.get_variable('weight',[self.num_neuron, self.num_classes], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.num_neuron)))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_classes, ]))

        with tf.variable_scope('loss') as scope:
            # cross-entropy 모델을 설정한다.
            # y_output = tf.nn.softmax(tf.matmul(y, weight) + bias)
            y_output = tf.matmul(y, weight) + bias
            y_output = tf.expand_dims(y_output, 0)
            self.y = y_output

            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.ground_label)
            self.cross_entropy = tf.reduce_mean(self.cross_entropy)
            # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ground_label * tf.log(self.y), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer().minimize(self.cross_entropy)

            tf.summary.scalar('loss', self.cross_entropy)

    def add_pred_op(self):
        self.labels_pred = tf.cast(tf.argmax(self.y, axis=-1), tf.int32)

    def build(self):
        self.add_placeholders()
        self.add_logits_op()
        self.add_pred_op()


    def run_epoch(self, sess, train_data, dev_data, test_data, epoch):
        """
        :param train_data: contains concatenated sentence(user and system list type) and ground_labels(O, T, X)
        :return: accuracy and f1 scroe
        """
        num_batches = (len(train_data) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=num_batches)

        for i, (concat_utter_list, ground_label) in enumerate(minibatches(train_data + dev_data + test_data[:300], self.config.batch_size)):
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

            ground_label_list = []
            for label in ground_label:
                # label.strip().encode('utf-8')
                ground_label_list.append(self.cate_mapping_dict[label.strip()])
            ground_label_list = np.array([ground_label_list])

            feed_dict = {
                self.input_features: input_features,
                self.ground_label: ground_label_list
            }
            self.merged = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)

            _, train_loss, summary = sess.run([self.train_step, self.cross_entropy, self.merged], feed_dict=feed_dict)

            prog.update(i + 1, [("train loss", train_loss)])

            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * num_batches + i)

            # accuracy, f1_score = self.run_evaluate(sess, test_data[300:])
            # self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * accuracy, 100 * f1_score))
            # return accuracy, f1_score

    def run_evaluate(self, sess, test_data):
        # create confusion matrix to evaluate precision and recall
        confusion_matrix = np.zeros(shape=(3, 3))

        accuracy_list = []
        for i, (concat_utter_list, ground_label) in enumerate(minibatches(test_data, self.config.batch_size)):
            input_features = []
            for each_utter_list in concat_utter_list:
                user_sentence = each_utter_list[0]
                system_sentence = each_utter_list[1]
                if self.config.embed_method == 'word2vec':
                    user_embedding = self.utter_embed.embed_utterance(user_sentence)
                    system_embedding = self.utter_embed.embed_utterance(system_sentence)
                    input_feature = np.concatenate((user_embedding, system_embedding))
                    input_features.append(input_feature)

            if self.config.embed_method == 'word2vec':
                input_features = np.array([input_features])

            ground_label_list = []
            for label in ground_label:
                ground_label_list.append(self.cate_mapping_dict[label.strip()])
            ground_label_list = np.array([ground_label_list])

            feed_dict = {
                self.input_features: input_features,
                # self.dropout_keep_prob: 1.0
            }

            labels_pred = sess.run([self.labels_pred], feed_dict=feed_dict)

            predict_list = list(labels_pred)[0][0]
            ground_list = ground_label_list[0]

            correct_pred = 0.
            for pred_ele, ground_ele in zip(predict_list, ground_list):
                confusion_matrix[pred_ele][ground_ele] += 1
                if pred_ele == ground_ele:
                    correct_pred += 1
                else:
                    continue
            accuracy_list.append(correct_pred / len(ground_list))
        accuracy = np.mean(accuracy_list)

        tp = 0.
        fp = 0.
        fn = 0.
        for i in range(3):
            tp += confusion_matrix[i][i]
            fp += (sum(confusion_matrix[:][i]) - confusion_matrix[i][i])
            fn += (sum(confusion_matrix[i][:]) - confusion_matrix[i][i])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)

        return accuracy, f1_score


    def train(self, train_data, dev_data, test_data):
        best_score = 0
        nepoch_no_imprv = 0

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # variables need to be initialized before we can use them
            sess.run(tf.global_variables_initializer())

            for step in range(10):
                # accuracy, f1_score = self.run_epoch(sess, train_data, dev_data, test_data, step)
                print(self.run_epoch(sess, train_data, dev_data, test_data, step))

                # add for early stopping
                # if f1_score >= best_score:
                #     nepoch_no_imprv = 0
                #     if not os.path.exists(self.config.model_output):
                #         os.makedirs(self.config.model_output)
                #     saver.save(sess, self.config.model_output)
                #     best_score = f1_score
                #     self.logger.info("- new best score!")
                #
                # else:
                #     nepoch_no_imprv += 1
                #     if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                #         self.logger.info("- early stopping {} epochs without improvement".format(
                #             nepoch_no_imprv))
                #         break



            # # 학습된 모델이 얼마나 정확한지를 출력한다.
            # correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.ground_label, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(sess.run(accuracy, feed_dict={self.input_features: self.input_features, self.ground_label: test_data[300:]}))

