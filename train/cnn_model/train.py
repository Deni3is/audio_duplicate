import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn(input_shape=(128, 128, 1), embedding_dim=512):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(embedding_dim, activation='relu'))  

    return model

# cnn_model = create_cnn()
# cnn_model.summary()

# # обучение 
# #TODO
# X_train,y_train = None,None
# X_val,y_val = None,None


# history = cnn_model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=10,
#     batch_size=32
# )
