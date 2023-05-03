from kerastuner import HyperModel
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation, LSTM, Reshape
import os
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


class CNNHypermodel(HyperModel):
    def __init__(self, input_shape, num_classes, mode):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.mode = mode

    def build(self, hp):

        # Hyperparameters
        hp_rate_1 = hp.Float(
            'dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05,
        )
        hp_rate_2 = hp.Float(
            'dropout_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05,
        )
        hp_rate_3 = hp.Float(
            'dropout_3', min_value=0.0, max_value=0.5, default=0.25, step=0.05,
        )
        hp_rate_4 = hp.Float(
            'dropout_4', min_value=0.0, max_value=0.5, default=0.25, step=0.05,
        )
        hp_rate_5 = hp.Float(
            'dropout_5', min_value=0.0, max_value=0.5, default=0.25, step=0.05,
        )
        hp_units_0 = hp.Int(
            'units_0', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_1 = hp.Int(
            'units_1', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_2 = hp.Int(
            'units_2', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_3 = hp.Int(
            'units_3', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_4 = hp.Int(
            'units_4', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_5 = hp.Int(
            'units_5', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_6 = hp.Int(
            'units_6', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_7 = hp.Int(
            'units_7', min_value=16, max_value=256, step=16, default=32
        )
        hp_units_8 = hp.Int(
            'units_8', min_value=16, max_value=256, step=16, default=32
        )
        #hp_units_9 = hp.Int(
        #    'units_9', min_value=16, max_value=256, step=16, default=32
        #)
        #hp_units_10 = hp.Int(
        #    'units_10', min_value=16, max_value=256, step=16, default=32
        #)
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4])
        # hp_activation = hp.Choice(
        #    'activation',
        #    values=['relu', 'tanh', 'sigmoid'],
        #    default='relu',
        # )

        # input
        inputs = Input(shape=(self.input_shape))

        # First Block
        model = Conv1D(hp_units_0, 8, padding='same')(inputs)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling1D(pool_size=(2))(model)
        model = Dropout(rate=hp_rate_1)(model)

        # Second Block
        model = Conv1D(hp_units_1, 8, padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling1D(pool_size=(2))(model)
        model = Dropout(rate=hp_rate_2)(model)

        # Third Block
        model = Conv1D(hp_units_2, 8, padding='same')(model)
        model = Activation('relu')(model)

        # Fourth Block
        model = Conv1D(hp_units_3, 8, padding='same')(model)
        model = Activation('relu')(model)

        # Fifth Block
        model = Conv1D(hp_units_4, 8, padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling1D(pool_size=(4))(model)
        model = Dropout(rate=hp_rate_3)(model)
        model = MaxPooling1D(pool_size=(4))(model)
        model = Dropout(rate=hp_rate_4)(model)

        # Sixth Block
        model = Conv1D(hp_units_5, 8, padding='same')(model)
        model = Activation('relu')(model)
        model = Dropout(rate=hp_rate_5)(model)

        model = Conv1D(hp_units_6, 8, padding='same')(model)
        model = Activation('relu')(model)

        model = Flatten()(model)
        #model = Reshape((1, model.shape[1]))(model)

        #model = LSTM(hp_units_9, return_sequences=True)(model)
        #model = LSTM(hp_units_10, return_sequences=True)(model)

        model = Dense(units=hp_units_7)(model)
        model = Activation('relu')(model)

        model = Dense(units=hp_units_8)(model)
        model = Activation('relu')(model)

        model = Dense(self.num_classes)(model)
        model = Activation('softmax')(model)

        loss = SparseCategoricalCrossentropy()
        # optimizer = Adam(learning_rate=LEARNING_RATE)
        # optimizer = RMSprop(learning_rate=LEARNING_RATE)

        model = Model(inputs=inputs, outputs=model)

        model.compile(loss=loss, optimizer=Adam(learning_rate=hp_learning_rate),
                      metrics=['accuracy'])

        return model
