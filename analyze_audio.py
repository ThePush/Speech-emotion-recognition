import os
import sys
import numpy as np
import librosa
import keras
from tqdm import tqdm
import os

# model parameters
MODEL = 'models/model_RAVDESS_CREMA_TESS_OG_CNN_hypermodel_fulltune.ckpt'
OBSERVED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# librosa parameters
AUDIO_PATH = sys.argv[1] if len(
    sys.argv) > 1 else 'audio.wav'
DURATION = 2
SAMPLE_RATE = 44100
N_MFCC = 40


def audio_to_features(path: str) -> np.ndarray:
    # Load the audio file
    y, sr = librosa.load(path)
    # Calculate the number of samples per chunk
    samples_per_chunk = int(sr * DURATION)
    # Divide the audio into chunks
    audio_chunks = [y[i:i+samples_per_chunk]
                    for i in range(0, len(y), samples_per_chunk)]

    # Extract features for each chunk (mfcc, chroma, mel) and store them in a numpy array of shape (n_chunks, n_features)
    features = np.array([])
    print('Extracting features...')
    for chunk in tqdm(audio_chunks, total=len(audio_chunks)):
        mfcc = np.mean(librosa.feature.mfcc(
            y=chunk, sr=sr, n_mfcc=N_MFCC).T, axis=0)
        stft = np.abs(librosa.stft(chunk))
        chroma = np.mean(librosa.feature.chroma_stft(
            S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(
            y=chunk, sr=sr).T, axis=0)
        ext_features = np.hstack([mfcc, chroma, mel])
        features = np.vstack([features, ext_features]
                             ) if features.size else ext_features
    
    print(f'Extracted {features.shape[0]} features from {len(audio_chunks)} chunks\n')

    return features


def predict_emotion(features: np.ndarray) -> None:
    model = keras.models.load_model(MODEL)
    print('\nPredicting emotion...')
    for i, chunk in enumerate(features):
        chunk = chunk[np.newaxis, ...]
        prediction = model.predict(chunk)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_emotion = OBSERVED_EMOTIONS[predicted_index[0]]
        print(
            f'Predicted emotion for chunk {i*DURATION}-{(i+1)*DURATION}s: {predicted_emotion}')
        print(f'Confidence: {prediction[0][predicted_index[0]]*100:.2f}%')


def main():
    print('\n' + '*' * 80)
    print('Analyzing audio from source: ', AUDIO_PATH)
    predict_emotion(audio_to_features(AUDIO_PATH))


if __name__ == '__main__':
    main()
