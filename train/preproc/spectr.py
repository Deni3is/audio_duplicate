
import os
import numpy as np
np.complex = complex  
import librosa
import tensorflow as tf

SAMPLE_RATE = 16000
MEL_SIZE = 128
SEGMENT_DURATION = 10.0 
OVERLAP = 3.0            
SUBSEGMENT_DURATION = 1.0  

# === 1. Нарезка аудио на перекрывающиеся 10-секундные фрагменты ===
def audio_to_overlapping_chunks(filepath, sr=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, overlap=OVERLAP):
    y, sr = librosa.load(filepath, sr=sr)
    segment_len = int(segment_duration * sr)
    step = int((segment_duration - overlap) * sr)

    chunks = []
    for start in range(0, len(y) - segment_len + 1, step):
        chunk = y[start:start + segment_len]
        chunks.append(chunk)

    return chunks

# === 2. Преобразование 10-секундного фрагмента в 10-секундные мел-спектрограммы ===
def chunk_to_melspec_sequence(chunk, sr=SAMPLE_RATE, segment_duration=SUBSEGMENT_DURATION, mel_size=MEL_SIZE):
    segment_len = int(segment_duration * sr)
    mels = []

    for i in range(0, len(chunk), segment_len):
        seg = chunk[i:i + segment_len]
        if len(seg) < segment_len:
            break

        mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=mel_size)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
        mel_db = mel_db[:128, :]
        mel_db = mel_db / np.max(np.abs(mel_db))
        mel_db = np.expand_dims(mel_db, axis=-1)
        mels.append(mel_db)

    return np.array(mels)  

# === 3. Генерация последовательностей эмбеддингов через CNN ===
def process_audio_to_embedding_sequences(filepath, cnn_model, output_dir="cnn_embeddings", custom_name=None):
    os.makedirs(output_dir, exist_ok=True)
    filename = custom_name if custom_name else os.path.splitext(os.path.basename(filepath))[0]

    chunks = audio_to_overlapping_chunks(filepath)
    all_embeddings = []

    for chunk in chunks:
        mel_batch = chunk_to_melspec_sequence(chunk)  # → (N, F, T)
        if mel_batch.shape[0] == 0:
            continue

        embeddings = cnn_model.predict(mel_batch, verbose=0)  # → (N, 512)
        all_embeddings.extend(embeddings)  # ✅ добавляем поштучно, а не список

    if not all_embeddings:
        print(f"⚠ Пропущено: {filename}, пустой список")
        return

    out_array = np.array(all_embeddings)  # → shape (T, 512)
    output_path = os.path.join(output_dir, f"{filename}_embeddings.npy")
    np.save(output_path, out_array)
    print(f"✓ Сохранено: {output_path}, shape = {out_array.shape}")
    return output_path

