import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model


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
        self.model = Model(inputs, outputs, name='rps_predictor')
        self.model.summary()
        keras.utils.plot_model(self.model, 'rps_predictor.png', show_shapes=True)

    def load(self, file):
        """
        Load pre-trained keras model. Stores instance of the
        model as an attribute of the class.

        Args:
            file (str):   Path to keras (.h5) model file.

        """

        self.model = load_model(file)

    def train(self, X, y):
        """
        Re-train keras model for a new dataset of simulated
        galaxy images and corresponding annotations.

        Args:
            X (np.array):   Array of images of shape (N_images, `self.input_shape`).
            y (np.array):   Annotations corresponding to training images X.

        Returns:
            model (??):     Trained keras model instance.
        """

    def predict(self, X):
        """
        Runs inference through the model that has been loaded.

        Args:
            X:      Images to run inference over with the trained CNN model
                    (self.model).

        Returns:
            y_p:    Array of predictions of RP parameter from input images X.
        """

        return self.model.predict(X)

    def mean_activation(self, X, layer=2):
        """

        """
        pass

    def architecture(self):
        """
        Print a summary of the CNN model architecture to the screen.
        """

        self.model.summary()
