
import os
import numpy as np
np.complex = complex  
import librosa
import soundfile as sf
import random
from pydub import AudioSegment

INPUT_DIR = r"D:\music"
OUTPUT_DIR = r"D:\music\train"
DURATION = 30  
SAMPLE_RATE = 16000
NOISE_LEVELS = [0.005, 0.1, 0.02]  


def add_noise(y, noise_level):
    noise = np.random.randn(len(y)) * noise_level
    return np.clip(y + noise, -1, 1)

def random_cut(y, sr, duration=30):
    total_len = len(y)
    target_len = int(sr * duration)
    if total_len <= target_len:
        return y
    start = random.randint(0, total_len - target_len)
    return y[start:start + target_len]

def convert_to_wav_if_needed(filepath):
    if filepath.lower().endswith(".mp3"):
        sound = AudioSegment.from_mp3(filepath)
        wav_path = filepath[:-4] + "_converted.wav"
        sound.export(wav_path, format="wav")
        return wav_path
    return filepath

def apply_echo(audioseg, delay_ms=250, attenuation_db=6):
    echo = audioseg.overlay(audioseg - attenuation_db, position=delay_ms)
    return echo

def apply_reverse(audioseg):
    return audioseg.reverse()

os.makedirs(OUTPUT_DIR, exist_ok=True)
track_id = 1

for fname in os.listdir(INPUT_DIR):
    if not (fname.lower().endswith(".wav") or fname.lower().endswith(".mp3")):
        continue

    fpath = os.path.join(INPUT_DIR, fname)
    wav_path = convert_to_wav_if_needed(fpath)
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    if len(y) < sr * DURATION:
        continue

    orig = y[:sr * DURATION]
    cut = random_cut(y, sr, 10)

    track_folder = os.path.join(OUTPUT_DIR, f"track{track_id:04d}")
    os.makedirs(track_folder, exist_ok=True)

    sf.write(os.path.join(track_folder, "original.wav"), orig, sr)
    sf.write(os.path.join(track_folder, "cut.wav"), cut, sr)

    for i, level in enumerate(NOISE_LEVELS):
        noise = add_noise(orig, level)
        sf.write(os.path.join(track_folder, f"noise_{i+1}.wav"), noise, sr)

    audio_seg = AudioSegment.from_wav(wav_path)[:DURATION * 1000]
    echo = apply_echo(audio_seg)
    reverse = apply_reverse(audio_seg)

    echo.export(os.path.join(track_folder, "echo.wav"), format="wav")
    reverse.export(os.path.join(track_folder, "reverse.wav"), format="wav")

    track_id += 1
    print("запись готова", fname)

print(f"✓ Подготовлено {track_id - 1} треков с шумами, эхом и реверсом в {OUTPUT_DIR}")
