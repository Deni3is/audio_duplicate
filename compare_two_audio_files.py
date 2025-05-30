import numpy as np
from tensorflow.keras.models import load_model
import os

# === Импорт локальных модулей ===
from train.preproc.spectr import audio_to_overlapping_chunks, chunk_to_melspec_sequence
from train.cnn_model.train import create_cnn
from train.tsn_model.train import create_tsn, TemporalShiftLayer
from train.siamse_model.train import create_siamese_network, euclidean_distance

# === Пути к моделям и аудиофайлам ===
WAV_1 = r"D:\music\val\track0651\original.wav"
WAV_2 = r"D:\music\val\track0620\original.wav"

# === Функция извлечения TSN-эмбеддинга для одного аудиофайла ===
def extract_embedding(filepath, cnn_model, tsn_model):
    chunks = audio_to_overlapping_chunks(filepath)
    all_embeddings = []

    for chunk in chunks:
        mel_batch = chunk_to_melspec_sequence(chunk)
        if mel_batch.shape[0] == 0:
            continue
        emb = cnn_model.predict(mel_batch, verbose=0)
        all_embeddings.extend(emb)

    if not all_embeddings:
        print(f"⚠️ Пустые эмбеддинги для {filepath}")
        return None

    cnn_seq = np.array(all_embeddings)  # (T, 512)

    # ⬇️ Приведение к форме (10, 512), как требует TSN
    if cnn_seq.shape[0] > 10:
        cnn_seq = cnn_seq[:10]
    elif cnn_seq.shape[0] < 10:
        pad = np.zeros((10 - cnn_seq.shape[0], 512))
        cnn_seq = np.concatenate([cnn_seq, pad], axis=0)

    tsn_input = np.expand_dims(cnn_seq, axis=0)  # (1, 10, 512)
    tsn_output = tsn_model.predict(tsn_input, verbose=0)[0]  # (256,)
    return tsn_output

# === Загрузка моделей ===
print("Загрузка моделей...")
cnn_model = create_cnn()
# cnn_model.load_weights("models/cnn_weights.h5")  # если веса сохранялись отдельно

tsn_model = load_model(
    r"C:\Users\levsh\Desktop\диплом\audio_duplicate\models\tsn_model.h5",
    custom_objects={"TemporalShiftLayer": TemporalShiftLayer}
)

siamese_model = load_model(
    r"C:\Users\levsh\Desktop\диплом\audio_duplicate\models\siamese_model.h5",
    custom_objects={"euclidean_distance": euclidean_distance}
)

# === Извлечение эмбеддингов ===
print(f"Извлечение признаков из:\n - {WAV_1}\n - {WAV_2}")
emb1 = extract_embedding(WAV_1, cnn_model, tsn_model)
emb2 = extract_embedding(WAV_2, cnn_model, tsn_model)

if emb1 is None or emb2 is None:
    print("❌ Не удалось извлечь эмбеддинги для одной из записей.")
    exit()

# === Сравнение через сиамскую сеть ===
print("Сравнение записей...")
similarity = siamese_model.predict([np.array([emb1]), np.array([emb2])])[0][0]
print(f"\n🎯 Степень схожести: {similarity:.4f}")

if similarity > 0.5:
    print("💡 Вероятно, это нечеткие дубликаты.")
else:
    print("🚫 Записи скорее всего разные.")
