from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model


class RPSPredictor():
    def __init__(self, n_maps, n_variables):
        self.model = None
        self.n_maps = n_maps
        self.n_variables = n_variables
        self.input_shape = (50, 50, n_maps)

    def load_model(self, path, method="model"):
        """
        Loads a model from a specified path that contains the model
        and weights files (.h5 format). Method can be either "model"
        or "weights".
        """
        assert (method == "model") or (
            method == "weights"), "Invalid model read method."
        if method == "model":
            self.model = load_model(path)
        if method == "weights":
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3),
                                  activation='relu',
                                  input_shape=self.input_shape))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.n_variables, activation='linear'))
            self.model.load_weights(path)

    def predict(self, X):
        """
        Runs inference through the model that has been loaded.
        """
        return self.model.predict(X)
