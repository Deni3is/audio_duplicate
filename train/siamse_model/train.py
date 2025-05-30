import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# Функция расстояния
@tf.keras.utils.register_keras_serializable() 
def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# Общая ветвь (Shared network)
def build_shared_network(input_shape=(256,), drop_rate=0.1):
    model = models.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(32, activation='relu')
    ])
    return model

# Построение сиамской модели
def create_siamese_network(input_shape=(256,)):
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    shared_network = build_shared_network(input_shape)

    encoded_a = shared_network(input_a)
    encoded_b = shared_network(input_b)

    distance = layers.Lambda(euclidean_distance)([encoded_a, encoded_b])
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = models.Model(inputs=[input_a, input_b], outputs=output)
    return model



# model = create_siamese_network()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Пример обучения:
# # X_train = (embeddings_a, embeddings_b)
# # y_train = [0, 1, 1, 0, ...]
# # embeddings_a, embeddings_b = np arrays with shape (batch, 256)

# model.fit([X_train_a, X_train_b], y_train, epochs=10, batch_size=32)
