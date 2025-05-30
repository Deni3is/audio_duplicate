import tensorflow as tf
from tensorflow.keras import layers, models

class TemporalShiftLayer(tf.keras.layers.Layer):
    def __init__(self, shift_ratio=0.25,**kwargs):
        super().__init__(**kwargs)  # обязательно передаём все стандартные аргументы
        self.shift_ratio = shift_ratio

    def call(self, x):
        B, T, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        shift_ratio_tensor = tf.constant(self.shift_ratio, dtype=tf.float32)
        shift_float = tf.cast(T, tf.float32) * shift_ratio_tensor
        shift = tf.cast(shift_float, tf.int32)
        x_left = tf.roll(x, shift=-shift, axis=1)
        x_right = tf.roll(x, shift=shift, axis=1)
        return (x + x_left + x_right) / 3.0

# Создание TSN-модели
def create_tsn(input_shape=(10, 512), output_dim=256):
    inputs = layers.Input(shape=input_shape)
    x = TemporalShiftLayer()(inputs)                      
    x = layers.Bidirectional(layers.LSTM(128))(x)         
    x = layers.Dense(output_dim, activation='relu')(x)    
    model = models.Model(inputs, x)
    return model




# # Создание модели
# tsn_model = create_tsn()

# # Пример входа: (batch_size, 10, 512)
# dummy_input = tf.random.normal((4, 10, 512))
# output = tsn_model(dummy_input)

# print("Выходной размер:", output.shape)  


# x = layers.Dense(1, activation='sigmoid')(x)  # бинарный классификатор

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10)
