import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from DataGenerator import DataGenerator


class DeepHedging(keras.Model):
    def __init__(self, dist_size, data_train, data_val):
        super(DeepHedging, self).__init__()
        self.dist_size = dist_size
        self.delta_hedge = self.create_model()
        self.data_train = data_train
        self.data_val = data_val
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_tracker = keras.metrics.Mean(name="loss", dtype=tf.float64)
        self.vtimestep_Euler = np.linspace(0, 1, dist_size+1)
        self.price = tf.Variable(0.0, dtype=tf.float64)
        self.trainable_variables.append(self.price)
        self.history = {
            "loss_train" : list(),
            "loss_val" : list()
        }

    def create_model(self):
        '''
        creates model used as delta hedge strategy
        '''

        state = Input(shape=(None, 2), dtype=tf.float64)
        layer = Dense(units=8, activation=tf.keras.activations.swish, dtype=tf.float64)(state)
        layer = Dense(units=8, activation=tf.keras.activations.swish, dtype=tf.float64)(layer)
        layer = Dense(units=8, activation=tf.keras.activations.swish, dtype=tf.float64)(layer)
        output = Dense(units=1, dtype=tf.float64)(layer)
        model = tf.keras.Model(inputs=state, outputs=output)

        return model

    def payoff(self, x, K=1.):
        return tf.where(x>K, x-K, 0.0)

    def loss_function(self, X, batch_size):
        '''
        computes loss function associated to deep hedging problem
        '''

        time = tf.tile(self.vtimestep_Euler[tf.newaxis,:], [batch_size,1])
        state = tf.concat([time[:,:-1,tf.newaxis],X[:,:-1,tf.newaxis]], axis=2)
        strategy_pnl = tf.reduce_sum(self.delta_hedge(state)[:,:,0] * tf.experimental.numpy.diff(X, n=1, axis=-1), axis=1)

        return tf.reduce_mean(tf.pow(self.price + strategy_pnl - self.payoff(X[:,-1]), 2))

    def train_delta_strategy(self, epochs, batch_size):
        '''
        training loop
        @params:
        - epochs: integer, number of epochs
        - batch_size, integer, mini-batch size
        '''

        # data generator for training dataset
        data_generator = DataGenerator(self.data_train, self.dist_size, batch_size, shuffle=True)

        # best validation loss and best NN weights
        best_loss_val = np.inf

        # training loop
        for epoch in range(epochs):

            print("\nepoch {}/{}".format(epoch+1,epochs))
            prog_bar = tf.keras.utils.Progbar(data_generator.__len__(), stateful_metrics=None)

            for index in range(data_generator.__len__()):
                X = data_generator.__getitem__(index, self.dist_size)
                with tf.GradientTape() as tape:
                    loss = self.loss_function(X, batch_size)

                # Compute gradients
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)

                # Update weights
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                # update tracker
                self.loss_tracker.update_state(loss)

                # update progress bar
                prog_bar.add(1, values=[("loss", self.loss_tracker.result())])

                # clear tracker at the end
                if epoch+1 == epochs:
                    self.loss_tracker.reset_states()
                    data_generator.on_epoch_end()

            
            loss_train = self.loss_function(self.data_train, self.data_train.shape[0])
            loss_val = self.loss_function(self.data_val, self.data_val.shape[0])
            self.history["loss_train"].append(loss_train)
            self.history["loss_val"].append(loss_val)

            print(f"loss_train = {loss_train} - loss_val = {loss_val}")

            if loss_val < best_loss_val:
                best_loss_val = loss_val.numpy()
                self.delta_hedge.save_weights("model.h5")

        print(self.trainable_variables[-1])
        self.delta_hedge.load_weights("model.h5")
        print(self.trainable_variables[-1])
        