import tensorflow as tf
import time
import os
import math
from data_process import minibatches
from general_utils import Progbar, print_sentence
import numpy as np


class Dnn():
    def __init__ (self, input_size, num_neuron, num_classes, utter_embed, config):
        self.input_size = input_size
        self.num_neuron = num_neuron
        self.num_classes = num_classes
        self.utter_embed = utter_embed
        self.logger = config.logger
        self.config = config

        self.cate_mapping_dict = {'O': 0, 'X': 1, 'T': 2}

    def add_placeholders(self):
        with tf.variable_scope('input') as scope:
            self.input_features = tf.placeholder(tf.float32, [1, 10, self.input_size], name='input_features')
            self.ground_label = tf.placeholder(tf.int32, [1, 10], name='ground_label')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def add_logits_op(self):
        with tf.variable_scope('hidden_layer1') as scope:
            reshaped_features = tf.transpose(self.input_features, [1, 0, 2])
            # print('reshaped_features: ', reshaped_features.shape)
            reshaped_features = tf.reshape(reshaped_features, [-1, self.input_size])

            # layer1
            weight_1 = tf.get_variable('weight', [self.input_size, self.num_neuron[0]], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.input_size)))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron[0], ]))
            # y1 = tf.matmul(x, W1) + b1
            z = tf.add(tf.matmul(reshaped_features, weight_1), bias)
            y = tf.nn.relu(z)
            y = tf.nn.dropout(y, self.dropout_keep_prob)

        name = ['hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6', 'hidden_layer7',
                'hidden_layer8', 'hidden_layer9']


        with tf.variable_scope(name[0]) as scope:
            # layer1
            weight_2 = tf.get_variable('weight', [self.num_neuron[0], self.num_neuron[0]],
                                     initializer=tf.random_normal_initializer(stddev=math.sqrt(2 / self.num_neuron[0])))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron[0], ]))
            # y1 = tf.matmul(x, W1) + b1
            y = tf.nn.relu(tf.add(tf.matmul(y, weight_2), bias))
            y = tf.nn.dropout(y, self.dropout_keep_prob)

        with tf.variable_scope(name[1]) as scope:
            # layer1
            weight_3 = tf.get_variable('weight', [self.num_neuron[0], self.num_neuron[1]],
                                     initializer=tf.random_normal_initializer(stddev=math.sqrt(2 / self.num_neuron[0])))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron[1], ]))
            # y1 = tf.matmul(x, W1) + b1
            y = tf.nn.relu(tf.add(tf.matmul(y, weight_3), bias))
            y = tf.nn.dropout(y, self.dropout_keep_prob)

        with tf.variable_scope(name[2]) as scope:
            # layer1
            weight_4 = tf.get_variable('weight', [self.num_neuron[1], self.num_neuron[2]],
                                     initializer=tf.random_normal_initializer(stddev=math.sqrt(2 / self.num_neuron[1])))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_neuron[2], ]))
            # y1 = tf.matmul(x, W1) + b1
            y = tf.nn.relu(tf.add(tf.matmul(y, weight_4), bias))
            y = tf.nn.dropout(y, self.dropout_keep_prob)

        with tf.variable_scope('output_layer') as scope:
            # layer2
            weight_out = tf.get_variable('weight',[self.num_neuron[2], self.num_classes], initializer=tf.random_normal_initializer(stddev=math.sqrt(2/self.num_neuron[2])))
            bias = tf.get_variable('bias', initializer=tf.zeros([self.num_classes, ]))
            y_output = tf.matmul(y, weight_out) + bias
            y_output = tf.expand_dims(y_output, 0)

        with tf.variable_scope('loss') as scope:
            self.y = y_output
            # 0.344 : 0.495 : 0.161
            classes_weights = tf.constant([0.346, 0.495, 0.161])
            self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=self.y,
                                                                     targets=tf.one_hot(self.ground_label, depth=3),
                                                                     pos_weight=classes_weights)
            # self.loss = tf.reduce_mean(cross_entropy)
            # self.cross_entropys = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.ground_label)
            # self.cross_entropy = tf.reduce_mean(self.cross_entropys)

        with tf.variable_scope('regularization') as scope:
            # weights = [weight_1, weight_2, weight_3, weight_4, weight_out]
            # for i in range(self.config.num_layer):
            #     regularizer = regularizer + tf.nn.l2_loss(weights[i])
            regularizer1 = tf.nn.l2_loss(weight_1)
            regularizer2 = tf.nn.l2_loss(weight_2)
            regularizer3 = tf.nn.l2_loss(weight_3)
            regularizer4 = tf.nn.l2_loss(weight_4)
            regularizer_out = tf.nn.l2_loss(weight_out)
            regularizer = regularizer1 + regularizer2 + regularizer3 + regularizer4 + regularizer_out

            self.cross_entropy = tf.reduce_mean(self.cross_entropy + self.config.beta * regularizer)

        with tf.variable_scope('optimizer') as scope:
            self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.cross_entropy)
            # self.train_step = tf.train.AdagradOptimizer(self.config.lr).minimize(self.cross_entropy)

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
                user_embedding = self.utter_embed.embed_utterance(user_sentence)
                system_embedding = self.utter_embed.embed_utterance(system_sentence)
                input_feature = np.concatenate((user_embedding, system_embedding), axis=0)
                input_features.append(input_feature)

            input_features = np.array([input_features])

            ground_label_list = []
            for label in ground_label:
                # label.strip().encode('utf-8')
                ground_label_list.append(self.cate_mapping_dict[label.strip()])

            ground_label_list = np.array([ground_label_list])

            dropout_keep_prob = 0.5
            feed_dict = {
                self.input_features: input_features,
                self.ground_label: ground_label_list,
                self.dropout_keep_prob : dropout_keep_prob
            }

            # self.merged = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)

            _, train_loss = sess.run([self.train_step, self.cross_entropy], feed_dict=feed_dict)

            prog.update(i + 1, [("train loss", train_loss)])

        accuracy, precision_X, recall_X, f1_score_X, precision_B_T, recall_B_T, f1_score_B_T = self.run_evaluate(sess, test_data[300:])

        self.logger.info("accuracy : {:f}".format(accuracy))
        self.logger.info("precision_X : {:f}".format(precision_X))
        self.logger.info("recall_X : {:f}".format(recall_X))
        self.logger.info("f1_score_X : {:f}".format(f1_score_X))

        self.logger.info("precision X + T : {:f}".format(precision_B_T))
        self.logger.info("recall X + T : {:f}".format(recall_B_T))
        self.logger.info("f1_score X + T : {:f}".format(f1_score_B_T))

        return accuracy, f1_score_X

    def run_evaluate(self, sess, test_data):
        # create confusion matrix to evaluate precision and recall
        confusion_matrix = np.zeros(shape=(3, 3))

        accuracy_list = []
        for i, (concat_utter_list, ground_label) in enumerate(
                minibatches(test_data, self.config.batch_size)):
            input_features = []

            for each_utter_list in concat_utter_list:
                user_sentence = each_utter_list[0]
                system_sentence = each_utter_list[1]

                user_words_embedding = self.utter_embed.embed_utterance(user_sentence)
                system_words_embedding = self.utter_embed.embed_utterance(system_sentence)

                input_feature = np.concatenate((user_words_embedding, system_words_embedding), axis=0)
                input_features.append(input_feature)

            input_x = np.array([input_features])

            ground_label_list = []
            for label in ground_label:
                ground_label_list.append(self.cate_mapping_dict[label.strip()])
            ground_label_list = np.array([ground_label_list])

            feed_dict = {
                self.input_features: input_x,
                self.dropout_keep_prob: 1.0
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

        # O : Not a breakdown, T : Possible breakdown, X : Breakdown
        tp_O = confusion_matrix[0][0]
        tp_X = confusion_matrix[1][1]
        tp_T = confusion_matrix[2][2]

        fp_O = (sum(confusion_matrix[:][0]) - confusion_matrix[0][0])
        fp_X = (sum(confusion_matrix[:][1]) - confusion_matrix[1][1])
        fp_T = (sum(confusion_matrix[:][2]) - confusion_matrix[2][2])

        fn_O = (sum(confusion_matrix[0][:]) - confusion_matrix[0][0])
        fn_X = (sum(confusion_matrix[1][:]) - confusion_matrix[1][1])
        fn_T = (sum(confusion_matrix[2][:]) - confusion_matrix[2][2])
        print(confusion_matrix)

        precision_X = tp_X / (tp_X + fp_X)
        recall_X = tp_X / (tp_X + fn_X)
        f1_score_X = (2 * precision_X * recall_X) / (precision_X + recall_X)

        precision_B_T = (tp_X + tp_T) / ((tp_X + fp_X) + (tp_T + fp_T))
        recall_B_T = (tp_T + tp_X) / ((tp_T + fn_T) + (tp_X + fn_X))
        f1_score_B_T = (2 * precision_B_T * recall_B_T) / (precision_B_T + recall_B_T)

        return accuracy, precision_X, recall_X, f1_score_X, precision_B_T, recall_B_T, f1_score_B_T

    def train(self, train_data, dev_data, test_data):
        best_score = 0
        nepoch_no_imprv = 0

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # variables need to be initialized before we can use them
            sess.run(tf.global_variables_initializer())

            if self.config.reload:
                self.logger.info("Reloading the latest trained model...")
                saver.restore(sess, self.config.model_output)
            # self.add_summary(sess)

            for epoch in range(self.config.num_epochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.num_epochs))
                accuracy, f1_score = self.run_epoch(sess, train_data, dev_data, test_data, epoch)

                # print('Step : ', epoch)
                # self.run_epoch(sess, train_data, dev_data, test_data, epoch)

                # decay learning rate
                self.config.lr *= self.config.lr_decay

                # add for early stopping
                if f1_score >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1_score
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                            nepoch_no_imprv))
                        break
