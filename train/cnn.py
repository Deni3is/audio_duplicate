from cnn_model.train import create_cnn

cnn_model = create_cnn()
cnn_model.load_weights(r"C:\Users\levsh\Desktop\диплом\audio_duplicate\models\cnn_weights.h5")