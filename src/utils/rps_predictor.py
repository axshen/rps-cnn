import os
import pickle
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow.keras.optimizers import Adadelta


class RPSPredictor():
    """
    Class to capture the necessary attributes and functions for
    using the CNN model for prediction of RPS parameters.

    Attributes:
        input_shape (tuple):    Shape of input map for CNN model
        output_shape (tuple):   Shape of output node of CNN.
        model (keras model):    The Keras trained CNN model instance
    """

    def __init__(self, n_params=1, n_channels=1):
        """
        The constructor for the RPSPredictor class.

        Parameters:
            n_params (int):     Number of ram pressure parameters (vrel/rho) to
                                predict. Value is n_params=1 for either v_rel or
                                rho_igm prediction, or can be n_params=2 if attempting
                                to predict both simultaneously.
            n_channels (int):   Number of channels to be used in the CNN model
                                image input. Value should be n_channels=1 for either
                                2D density or kinematic maps, or n_channel=2 for
                                two-channeled map predictions.
        """

        self.output_shape = (n_params)
        self.input_shape = (50, 50, n_channels)
        self.model = self.__construct_model()
        self.training_history = None

    def __construct_model(self):
        """
        Construct CNN model with tensorflow functional API.
        """

        inputs = keras.Input(shape=self.input_shape, name='img')
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.output_shape, activation='linear')(x)
        model = Model(inputs, outputs, name='rps_predictor')
        return model

    def load(self, file):
        """
        Load pre-trained keras model. Stores instance of the
        model as an attribute of the class.

        Args:
            file (str):   Path to saved tensorflow model.

        """

        self.model = load_model(file)

    def compile(self, **kwargs):
        optimizer = Adadelta()
        loss = 'mean_square_error'
        if 'optimizer' in kwargs:
            optimizer = kwargs.get('optimizer')
            del kwargs['optimizer']
        if 'loss' in kwargs:
            loss = kwargs.get('loss')
            del kwargs['loss']
        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)

    def train(self, X, y, X_val, y_val, batch_size=32, epochs=100):
        """
        Re-train keras model for a new dataset of simulated
        galaxy images and corresponding annotations. Saves the training history
        to a records file.

        Args:
            X (np.array):   Array (n, 50, 50, n_channels) of input images.
            y (np.array):   Array (n, n_params) of annotations.

        Returns:
            model:          Trained keras model instance.
        """

        self.training_history = self.model.fit(
            X, y,
            batch_size=batch_size, epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val)
        )

        write_file = os.getenv('TRAIN_HISTORY', os.path.join(os.path.dirname(__file__), '../records/history'))
        with open(write_file, 'wb') as f:
            pickle.dump(self.training_history.history, f)

    def predict(self, X):
        """
        Runs inference through the model that has been loaded.

        Args:
            X (np.array):  Array (n, 50, 50, n_channels) of images.

        Returns:
            y_p (np.array): Array (n, n_params) of predictions for RP parameter
                            from input images X.
        """

        return self.model.predict(X)

    def mean_activation(self, X, layer=2):
        """

        """
        pass

    def summary(self):
        """
        Print a summary of the CNN model architecture to the screen.
        """

        self.model.summary()

    def save_summary(self, filename='rps_predictor.png'):
        keras.utils.plot_model(self.model, filename, show_shapes=True)

    def save(self, filename):
        """
        Save model h5
        """

        self.model.save(filename)