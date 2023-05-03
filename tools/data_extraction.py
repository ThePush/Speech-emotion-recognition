import librosa
import os
import sys
import numpy as np
import glob
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Normalize

"""
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""

OPTION = '6'

EMOTIONS = {1: 'neutral', 2: 'neutral', 3: 'happy',
            4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprised'}


OBSERVED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']

USE_MFCC = True
USE_CHROMA = True
USE_MEL = True

N_MFCC = 40
SAMPLE_RATE = 44100


def augmentation(audio) -> np.ndarray:
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.1, max_rate=0.15, p=0.5),
        # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        # Shift(min_fraction=-0.2, max_fraction=0.2, p=0.5),
        Normalize(p=1)
    ])
    return augment(samples=audio, sample_rate=SAMPLE_RATE)


def load_audio_files(path: str, augment: bool = False) -> list:
    audios = []

    print('Loading audio files...')
    for file in tqdm(path, total=len(path)):
        audios.append(librosa.load(file, sr=SAMPLE_RATE)[0])

    if augment:
        print('Augmenting audio files...')
        audios = [augmentation(audio)
                  for audio in tqdm(audios, total=len(audios))]
    return audios


def audio_to_features(audio):
    features = np.array([])
    if USE_MFCC:
        features = np.hstack((features, np.mean(librosa.feature.mfcc(
            y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T, axis=0)))
    if USE_CHROMA:
        stft = np.abs(librosa.stft(audio))
        features = np.hstack((features, np.mean(
            librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE).T, axis=0)))
    if USE_MEL:
        features = np.hstack((features, np.mean(
            librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE).T, axis=0)))
    return features


def extract_ravdess_data(file_name: str, audio):
    parts = file_name.split('/')[-1].split('.')[0].split('-')
    emotion = EMOTIONS[int(parts[2])]
    if emotion not in OBSERVED_EMOTIONS:
        return None, None

    features = audio_to_features(audio)

    return features, OBSERVED_EMOTIONS.index(emotion)


def extract_crema_data(file_name: str, audio):
    emotions_dict = {'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust',
                     'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral'}
    parts = file_name.split('/')[-1].split('.')[0].split('_')

    emotion = emotions_dict[parts[2]]
    if emotion not in OBSERVED_EMOTIONS:
        return None, None

    features = audio_to_features(audio)

    return features, OBSERVED_EMOTIONS.index(emotion)


def extract_tess_data(file_name: str, audio):
    parts = file_name.split('/')[-1].split('.')[0].split('_')

    emotion = parts[2]
    if emotion not in OBSERVED_EMOTIONS:
        return None, None

    features = audio_to_features(audio)

    return features, OBSERVED_EMOTIONS.index(emotion)


def extract_dataset(dataset, extract_data_function):
    files = glob.glob('resources/' + dataset + '/**/*.wav', recursive=True)
    print('*' * 100)
    print('Processing ' + dataset + ' dataset:')
    audios = load_audio_files(files, augment=False)
    x, y = [], []
    print('Extracting features from audios...')
    for i in tqdm(range(len(files)), total=len(files)):
        file_name = files[i]
        audio = audios[i]
        features, label = extract_data_function(file_name, audio)
        if features is None or label is None:
            continue
        x.append(features)
        y.append(label)

    x = np.array(x)
    y = np.array(y)
    print('Saving features and labels...')
    if not os.path.exists('np_arrays/' + dataset + OPTION):
        os.makedirs('np_arrays/' + dataset + OPTION)
    np.save('np_arrays/' + dataset + OPTION +
            '/x_' + dataset + OPTION + '.npy', x)
    np.save('np_arrays/' + dataset + OPTION +
            '/y_' + dataset + OPTION + '.npy', y)
    print('Data saved in .npy format in np_arrays/' +
          dataset + OPTION + '/ folder.')
    print('*' * 100)


def main():
    if len(sys.argv) > 1:
        sys.exit()
    extract_dataset('RAVDESS', extract_ravdess_data)
    # extract_dataset('TESS', extract_tess_data)
    # extract_dataset('CREMA', extract_crema_data)
    # extract_dataset(savee, savee_audios, extract_savee_data)
    # extract_dataset('validation', extract_ravdess_data)


if __name__ == '__main__':
    main()
