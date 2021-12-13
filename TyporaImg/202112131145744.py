import tensorflow as tf
import math
import logging
import modules
logger = logging.getLogger("model")


class Model:
    def __get_option_list(self, options, item_type, norm_val):
        if isinstance(options, list):
            if len(options) == 1:
                return [options[0]]*self.num_sessions
            elif len(options) < self.num_sessions:
                return options + [norm_val]*(self.num_sessions-len(options))
            else:
                return options
        elif isinstance(options, item_type):
            return [options] * self.num_sessions

    def __init__(self, node_counts, restore_path=None, **kwargs):
        self.l2 = kwargs.get('l2', 1e-5)
        self.lr = kwargs.get('lr', 0.001)
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.var_init = 1.0 / math.sqrt(self.hidden_size)
        self.node_counts = node_counts
        self.num_sessions = len(node_counts)
        self.use_norm = kwargs.get('use_norm', False)
        if kwargs.get('sigma', None) is not None:
            self.sigma = self.__get_option_list(kwargs.get('sigma', 16.0), int, 16)
        elif self.use_norm:
            self.sigma = tf.get_variable("sigma", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(16))

        self.node_embeddings = []
        self.ggnns = []
        self.node_aggregators = []
        self.session_transforms = []
        self.space_transforms = []

        self.batch_size = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.adj_ins = []
        self.adj_outs = []
        self.graph_items = []
        self.last_item_node_ids = []
        self.next_items = []

        gru_steps = self.__get_option_list(kwargs.get('gru_steps', 1), int, 1)
        node_aggregator_types = self.__get_option_list(kwargs.get('node_aggregator', 'self_attention'), str,
                                                       'self_attention')

        for i, node_count in enumerate(node_counts):
            self.node_embeddings.append(tf.get_variable("node_embedding_{}".format(i),
                                                            shape=[node_count, self.hidden_size], dtype=tf.float32,
                                                            initializer=tf.truncated_normal_initializer(0, self.var_init)))
            self.next_items.append(tf.placeholder(tf.int32, name="next_item_{}".format(i)))  # batch_size
            self.adj_ins.append(tf.placeholder(tf.float32, name="adj_in_{}".format(i)))  # batch_size, None, None
            self.adj_outs.append(tf.placeholder(tf.float32, name="adj_out_{}".format(i)))  # batch_size, None, None
            self.graph_items.append(tf.placeholder(tf.int32, name="graph_items_{}".format(i)))  # batch_size, None
            self.last_item_node_ids.append(tf.placeholder(tf.int32, name="last_item_node_id_{}".format(i)))  # batch_size
            self.ggnns.append(modules.GGNN(self.hidden_size, self.batch_size, gru_steps[i], residual_connect=kwargs.get('ggnn_residual', False)))
            node_agg_class = modules.get_node_aggregator(node_aggregator_types[i])
            if node_aggregator_types[i] == 'self_attention':
                self.node_aggregators.append(node_agg_class(self.hidden_size, self.batch_size, kwargs.get('sab_count', 4)))
            else:
                self.node_aggregators.append(node_agg_class(self.hidden_size, self.batch_size))

        if self.num_sessions > 1:
            dense_activate = kwargs.get('dense_activate', None)
            dense_bias = kwargs.get('dense_bias', None)
            for _ in range(self.num_sessions):
                self.session_transforms.append(modules.Dense(self.num_sessions*self.hidden_size, self.hidden_size,
                                                             activate=dense_activate, bias=dense_bias))
            self.session_att = modules.SessionSoftAttention(self.num_sessions, hidden_size=self.hidden_size)

        self.loss, self.session_state, self.logits = self.__forward()

        self.global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver()
        params = tf.trainable_variables()
        with open('vars', 'w') as f:
            for p in params:
                f.write(str(p)+'\n')

        lr_dc = kwargs.get('lr_dc', None)
        dc_rate = kwargs.get('dc_rate', 0.9)
        self.learning_rate = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=lr_dc,
                                                        decay_rate=dc_rate, staircase=True) if lr_dc else self.lr
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.loss_train = self.loss + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2
        self.opt = opt.minimize(self.loss_train, global_step=self.global_step)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if restore_path is not None:
            try:
                self.saver.restore(self.sess, restore_path)
            except ValueError as ve:
                logger.info(ve)
            except:
                logger.info("Restore Failed")

    def __forward(self):
        states = []
        for i in range(self.num_sessions):
            node_state = tf.nn.embedding_lookup(self.node_embeddings[i], self.graph_items[i])
            #if self.use_norm:
            #    node_state = tf.nn.l2_normalize(node_state, 1)
            graph_items_mask = tf.cast(tf.greater(self.graph_items[i], tf.zeros_like(self.graph_items[i])), tf.float32)
            # batch_size*hidden_size
            node_state = self.ggnns[i](self.adj_ins[i], self.adj_outs[i], node_state)
            states.append(self.node_aggregators[i](node_state, graph_items_mask, self.last_item_node_ids[i], keep_prob=self.keep_prob))

        if self.num_sessions > 1:
            states_in = tf.concat(states, axis=-1)
            session_state = [self.session_transforms[i](states_in, keep_prob=self.keep_prob) + states[i] for i in range(self.num_sessions)]
        else:
            session_state = [states[0]]

        if self.num_sessions > 1:
            if self.use_norm:
                norm_ss = [tf.nn.l2_normalize(ss, 1) for ss in session_state]
                z = self.session_att(norm_ss, query=norm_ss[0]) + norm_ss[0]
            else:
                z = self.session_att(session_state, query=session_state[0]) + session_state[0]
            # z = self.session_att(session_state, query=session_state[0]) + session_state[0]
            f_states = [z] + session_state[1:]
            if self.use_norm:
                logits_0 = tf.matmul(tf.nn.l2_normalize(f_states[0], 1), tf.nn.l2_normalize(self.node_embeddings[0][1:], 1), transpose_b=True) * self.sigma
                logits = [tf.matmul(f_states[i], self.node_embeddings[i][1:], transpose_b=True)
                          for i in range(1, self.num_sessions)]
                logits.insert(0, logits_0)
            else:
                logits = [tf.matmul(f_states[i], self.node_embeddings[i][1:], transpose_b=True)
                          for i in range(self.num_sessions)]
            losses = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.next_items[i] - 1, logits=logits[i])) for i in range(self.num_sessions)]
            loss = losses[0] * self.num_sessions + tf.add_n(losses[1:])
            return loss, z, logits[0]
        else:
            if self.use_norm:
                logits = tf.matmul(tf.nn.l2_normalize(session_state[0], 1),
                                   tf.nn.l2_normalize(self.node_embeddings[0][1:], 1), transpose_b=True) * self.sigma
            else:
                logits = tf.matmul(session_state[0], self.node_embeddings[0][1:], transpose_b=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.next_items[0] - 1, logits=logits))
            return loss, session_state[0], logits

    def __build_feed(self, input_sessions, has_next_item=True):
        if len(input_sessions) != self.num_sessions:
            raise ValueError("Session Count Error: input: {}, model: {}".format(len(input_sessions), self.num_sessions))
        feed_dict = {}
        for i, session in enumerate(input_sessions):
            if has_next_item:
                adj_in, adj_out, graph_item, last_node_id, next_item = session
            else:
                adj_in, adj_out, graph_item, last_node_id = session
            feed_dict[self.adj_ins[i]] = adj_in
            feed_dict[self.adj_outs[i]] = adj_out
            feed_dict[self.graph_items[i]] = graph_item
            feed_dict[self.last_item_node_ids[i]] = last_node_id
            if has_next_item:
                feed_dict[self.next_items[i]] = next_item
        return feed_dict

    def run_train(self, input_sessions):
        """
        :param input_sessions: list, [(adj_in, adj_out, graph_item, last_node_id, next_item),..]
        :return: train_loss
        """
        feed_dict = {self.keep_prob: 0.5, self.batch_size: len(input_sessions[0][2]), **self.__build_feed(input_sessions)}
        _, train_loss = self.sess.run([self.opt, self.loss_train], feed_dict=feed_dict)
        return train_loss

    def run_eval(self, input_sessions):
        feed_dict = {self.keep_prob: 1.0, self.batch_size: len(input_sessions[0][2]),
                     **self.__build_feed(input_sessions)}
        return self.sess.run([self.loss, self.logits], feed_dict=feed_dict)

    def run_embedding(self):
        return self.sess.run(self.node_embeddings)

    def run_session_embedding(self, input_sessions):
        feed_dict = {self.keep_prob: 1.0, self.batch_size: len(input_sessions[0][2]),
                     **self.__build_feed(input_sessions)}
        return self.sess.run(self.session_state, feed_dict=feed_dict)

    def run_predict(self, input_sessions):
        feed_dict = {self.keep_prob: 1.0, self.batch_size: len(input_sessions[0][2]),
                     **self.__build_feed(input_sessions, False)}
        return self.sess.run(self.logits, feed_dict=feed_dict)

    def run_step(self):
        return self.sess.run(self.global_step)

    def save(self, path, global_step=None):
        self.saver.save(self.sess, path, global_step)



