import numpy as np
from tensorflow.keras.models import load_model

from train.preproc.spectr import audio_to_overlapping_chunks, chunk_to_melspec_sequence
from train.cnn_model.train import create_cnn
from train.tsn_model.train import TemporalShiftLayer
from train.siamse_model.train import euclidean_distance

# WAV_1 = r"C:\Users\levsh\Desktop\tracks\track0010\original.txt"
# WAV_2 = r"C:\Users\levsh\Desktop\tracks\track0010\reverse.wav"


class Model:
    def __init__(self):
        print("Загрузка моделей...")    
        self.cnn_model = create_cnn()
        self.tsn_model = load_model(
            r"C:\Users\levsh\Desktop\git\audio_duplicate\modelv2\tsn_model.h5",
            custom_objects={"TemporalShiftLayer": TemporalShiftLayer}
        )
        self.siamese_model = load_model(
            r"C:\Users\levsh\Desktop\git\audio_duplicate\modelv2\siamese_model.h5",
            custom_objects={"euclidean_distance": euclidean_distance}
        )
    
    def extract_embedding(self, filepath, cnn_model, tsn_model):
        chunks = audio_to_overlapping_chunks(filepath)
        all_embeddings = []

        for chunk in chunks:
            mel_batch = chunk_to_melspec_sequence(chunk)
            if mel_batch.shape[0] == 0:
                continue
            emb = cnn_model.predict(mel_batch, verbose=0)
            all_embeddings.extend(emb)

        if not all_embeddings:
            print(f"Пустые эмбеддинги для {filepath}")
            return None

        cnn_seq = np.array(all_embeddings) 

        if cnn_seq.shape[0] > 10:
            cnn_seq = cnn_seq[:10]
        elif cnn_seq.shape[0] < 10:
            pad = np.zeros((10 - cnn_seq.shape[0], 512))
            cnn_seq = np.concatenate([cnn_seq, pad], axis=0)

        tsn_input = np.expand_dims(cnn_seq, axis=0)  
        tsn_output = tsn_model.predict(tsn_input, verbose=0)[0]  
        return tsn_output

    def inference(self, path1, path2):
        similarity = []
        print(f"Извлечение признаков из:\n - {path1}\n - {path2}")
        try:
            emb1 = self.extract_embedding(path1, self.cnn_model, self.tsn_model)
            emb2 = self.extract_embedding(path2, self.cnn_model, self.tsn_model)
        except Exception as ex:
            print("Не удалось извлечь эмбеддинги для одной из записей.")
            return None

        print("Сравнение записей...")
        for _ in range(3):
            similarity.append(self.siamese_model.predict([np.array([emb1]), np.array([emb2])])[0][0])

        for i in range(3):
            print(f"\n Степень схожести: {similarity[i]:.4f}")
        return max(similarity)
