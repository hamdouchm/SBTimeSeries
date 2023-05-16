import numpy as np

class DataGenerator:
    'Generates data for Keras'
    def __init__(self, data, distSize, batch_size=32, shuffle=True):
        'Initialization'
        self.data = data
        self.distSize = distSize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = list(range(len(data)))
        self.on_epoch_end()

    def __len__(self):
        'Returns the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index, time_index):
        'Generates one mini-batch'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(indexes, time_index)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes, time_index):
        'Generates data containing batch_size samples'
        return self.data[np.array(indexes),:time_index+1]