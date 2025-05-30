
import os
import time
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
from preproc.spectr import process_audio_to_embedding_sequences
from tsn_model.train import create_tsn
from siamse_model.train import create_siamese_network
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# === ПАРАМЕТРЫ ===
DATASET_DIR = "D:\music"
EMB_DIR = "D:\music\embedings"
SPLITS = ["train", "val"]

def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# === 1. Извлечение эмбеддингов из всех аудиофайлов ===
def extract_all_embeddings(cnn_model):
    for split in SPLITS:
        log(f"Обработка раздела: {split}")
        split_dir = os.path.join(DATASET_DIR, split)
        out_dir = os.path.join(EMB_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        for track_folder in os.listdir(split_dir):
            full_path = os.path.join(split_dir, track_folder)
            for fname in os.listdir(full_path):
                fpath = os.path.join(full_path, fname)
                unique_name = f"{track_folder}_{os.path.splitext(fname)[0]}"
                process_audio_to_embedding_sequences(fpath, cnn_model, output_dir=out_dir, custom_name=unique_name)


def load_embeddings_and_process_tsn(embedding_dir, tsn_model):
    """
    Загружает эмбеддинги CNN, прогоняет их через TSN и возвращает
    список эмбеддингов и соответствующих меток

    embedding_dir: папка с .npy файлами (CNN embeddings)
    tsn_model: обученная модель TSN

    Возвращает: X_tsn, labels
    """
    X = []
    labels = []

    for fname in os.listdir(embedding_dir):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(embedding_dir, fname)
        emb = np.load(path)  # shape: (T, 512)
        if len(emb.shape) != 2 or emb.shape[0] < 1:
            print(f"{fname} → shape: {emb.shape}")
            continue

        # Добавляем измерение батча → (1, T, 512)
        emb_input = np.expand_dims(emb, axis=0)
        tsn_output = tsn_model.predict(emb_input, verbose=0)  # → shape (1, 256)
        X.append(tsn_output[0])

        # Метка: track ID из имени файла
        label = fname.split("_")[0]
        labels.append(label)

    return np.array(X), np.array(labels)




def load_embeddings_with_labels(embedding_dir):
    X = []
    labels = []
    filenames = []

    for fname in os.listdir(embedding_dir):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(embedding_dir, fname)
        emb = np.load(path)  # shape: (n_chunks, 10, 512) или (n, 512)
        for emb_vector in emb:
            X.append(np.mean(emb_vector, axis=0))  # приведение к (512,)
            track_id = fname.split("_")[0]
            labels.append(track_id)
            filenames.append(fname)
    
    return np.array(X), np.array(labels), filenames

# === 2. Загрузка эмбеддингов для TSN ===
def load_embeddings_for_tsn(split_path):
    X, y = [], []
    log(f"Загрузка эмбеддингов из {split_path}")
    for fname in os.listdir(split_path):
        path = os.path.join(split_path, fname)
        emb = np.load(path)  # (n, 10, 512)
        for segment in emb:
            X.append(segment)
            y.append(1)
    log(f"Загружено: {len(X)} примеров")
    return np.array(X), np.array(y)

def generate_balanced_siamese_pairs(embeddings, labels):
    X_a, X_b, y = [], [], []
    label_to_indices = {}

    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    # Положительные пары
    positive_pairs = []
    for indices in label_to_indices.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                positive_pairs.append((indices[i], indices[j]))

    # Отрицательные пары
    negative_pairs = []
    all_labels = list(label_to_indices.keys())
    while len(negative_pairs) < len(positive_pairs):
        l1, l2 = random.sample(all_labels, 2)
        i = random.choice(label_to_indices[l1])
        j = random.choice(label_to_indices[l2])
        negative_pairs.append((i, j))

    # Собираем X и y
    for i, j in positive_pairs:
        X_a.append(embeddings[i])
        X_b.append(embeddings[j])
        y.append(1)

    for i, j in negative_pairs:
        X_a.append(embeddings[i])
        X_b.append(embeddings[j])
        y.append(0)

    return np.array(X_a), np.array(X_b), np.array(y)


def generate_siamese_pairs_from_embeddings(embeddings, labels=None, num_negatives=0.5, max_pairs=10000):
    """
    embeddings: np.array, shape (N, D)
    labels: np.array of shape (N,) or None — идентификаторы классов, если есть
    num_negatives: сколько отрицательных пар на каждую положительную
    max_pairs: максимальное общее количество пар для избежания переполнения памяти

    Возвращает: X_a, X_b, y
    """
    X_a, X_b, y = [], [], []
    N = len(embeddings)
    count = 0

    if labels is None:
        # Предполагаем, что каждые 6 подряд — из одного класса
        group_size = 6
        for i in range(N):
            for j in range(i + 1, N):
                same = (i // group_size == j // group_size)
                X_a.append(embeddings[i])
                X_b.append(embeddings[j])
                y.append(1 if same else 0)
                count += 1
                if count >= max_pairs:
                    return np.array(X_a), np.array(X_b), np.array(y)
    else:
        # С использованием явных меток классов
        label_to_indices = {}
        for i, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(i)

        for label, pos_indices in label_to_indices.items():
            # Положительные пары
            for i in range(len(pos_indices)):
                for j in range(i + 1, len(pos_indices)):
                    X_a.append(embeddings[pos_indices[i]])
                    X_b.append(embeddings[pos_indices[j]])
                    y.append(1)
                    count += 1
                    if count >= max_pairs:
                        return np.array(X_a), np.array(X_b), np.array(y)

            # Отрицательные пары
            neg_labels = [l for l in label_to_indices if l != label]
            for i in pos_indices:
                for _ in range(num_negatives):
                    neg_label = random.choice(neg_labels)
                    j = random.choice(label_to_indices[neg_label])
                    X_a.append(embeddings[i])
                    X_b.append(embeddings[j])
                    y.append(0)
                    count += 1
                    if count >= max_pairs:
                        return np.array(X_a), np.array(X_b), np.array(y)

    return np.array(X_a), np.array(X_b), np.array(y)


# === 3. Генерация пар для сиамской сети ===
def generate_siamese_pairs(embedding_dir):
    pairs_a, pairs_b, labels = [], [], []
    log(f"Генерация пар из {embedding_dir}")
    folders = os.listdir(embedding_dir)
    for i, f1 in enumerate(folders):
        emb1 = np.load(os.path.join(embedding_dir, f1))
        mean_1 = [np.mean(e, axis=0) for e in emb1]
        for j in range(len(mean_1)):
            for k in range(j + 1, len(mean_1)):
                pairs_a.append(mean_1[j])
                pairs_b.append(mean_1[k])
                labels.append(1)
        for f2 in folders[i + 1:]:
            emb2 = np.load(os.path.join(embedding_dir, f2))
            mean_2 = [np.mean(e, axis=0) for e in emb2]
            for a in mean_1:
                for b in mean_2:
                    pairs_a.append(a)
                    pairs_b.append(b)
                    labels.append(0)
    log(f"Создано {len(labels)} пар")
    return np.array(pairs_a), np.array(pairs_b), np.array(labels)

# === 4. Обучение всей цепочки ===
def train_pipeline(cnn_model):
    t0 = time.time()
    log("1. Извлечение эмбеддингов...")
    # extract_all_embeddings(cnn_model)  # включи, если нужно извлекать заново
    log(f"✓ Эмбеддинги извлечены за {int(time.time() - t0)} сек")

    # === TSN ===
    log("2. Обработка эмбеддингов через TSN...")
    t1 = time.time()
    tsn_model = create_tsn()

    X_train_emb, y_train_labels = load_embeddings_and_process_tsn(os.path.join(EMB_DIR, "train"), tsn_model)
    X_val_emb, y_val_labels = load_embeddings_and_process_tsn(os.path.join(EMB_DIR, "val"), tsn_model)
    log(f"✓ TSN обработка завершена за {int(time.time() - t1)} сек")
    print(f"TSN → train: {X_train_emb.shape}, val: {X_val_emb.shape}")
    tsn_model.save("models/tsn_model.h5")
    log("📦 TSN модель сохранена в models/tsn_model.h5")
    # === Сиамская сеть ===
    log("3. Обучение сиамской сети...")
    t2 = time.time()

    X_a, X_b, y = generate_balanced_siamese_pairs(X_train_emb, y_train_labels)
    X_a_val, X_b_val, y_val = generate_balanced_siamese_pairs(X_val_emb, y_val_labels)

    print("X_a shape:", X_a.shape)
    print("X_b shape:", X_b.shape)
    print("y shape:", y.shape)

    siamese_model = create_siamese_network()
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = siamese_model.fit(
        [X_a, X_b], y,
        validation_data=([X_a_val, X_b_val], y_val),
        epochs=1000,
        batch_size=64*2
    )
    log(f"✓ Сиамская сеть обучена за {int(time.time() - t2)} сек")
    # === Сохранение модели ===
    siamese_model.save("models/siamese_model.h5")
    log("📦 Сиамская модель сохранена в models/siamese_model.h5")
    # === Графики ===
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig("loss_plot.png")

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("accuracy_plot.png")

    # === Оценка ===
    y_pred_prob = siamese_model.predict([X_a_val, X_b_val])
    y_pred = (y_pred_prob > 0.7).astype(int)

    print("== Classification Report ==")
    print(classification_report(y_val, y_pred))

    print("== Confusion Matrix ==")
    print(confusion_matrix(y_val, y_pred))

    print("== ROC AUC ==")
    print("AUC:", roc_auc_score(y_val, y_pred_prob))

    log(f"✅ Все этапы завершены за {int(time.time() - t0)} сек")


# === Пример запуска ===
if __name__ == "__main__":
    from cnn_model.train import create_cnn
    cnn = create_cnn()
    train_pipeline(cnn)
