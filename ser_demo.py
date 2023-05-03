import os
import sys
import numpy as np
import librosa
import keras
from tqdm import tqdm
import os
import time
from pydub import AudioSegment
from pydub.playback import play
import pytube
import io

# model parameters
MODEL = 'models/model_RAVDESS_CREMA_TESS_OG_CNN_hypermodel_fulltune.ckpt'
OBSERVED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# librosa parameters
DURATION = 3
SAMPLE_RATE = 44100
N_MFCC = 40


def download_audio(url: str) -> str:
    print(f'Downloading audio from \'{url}\'...')
    yt = pytube.YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()

    # Convert the audio stream to a byte string using pytube
    audio_data = io.BytesIO()
    audio_stream.stream_to_buffer(audio_data)
    audio_data.seek(0)

    # Convert the byte string to an AudioSegment using pydub
    audio_segment = AudioSegment.from_file(audio_data)

    # Save the audio segment as a .wav file
    audio_segment.export(yt.title + ".wav", format="wav")
    print('Video title: ', yt.title)
    print('Video length: ', yt.length, 'seconds')
    print('.wav size: ', os.path.getsize(yt.title + ".wav")/1000000, 'MB')
    print('Download complete\n')

    return yt.title


def audio_to_features(path: str) -> np.ndarray:
    print(f'Loading audio file: {path}...')
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

    print(
        f'Extracted {features.shape[0]} features from {len(audio_chunks)} chunks of {DURATION}s each\n')

    return features


def predict_emotion(features: np.ndarray, title: str) -> None:
    model = keras.models.load_model(MODEL)
    print('\nPredicting emotions...')
    for i, chunk in enumerate(features):
        chunk = chunk[np.newaxis, ...]
        prediction = model.predict(chunk)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_emotion = OBSERVED_EMOTIONS[predicted_index[0]]
        print(
            f'Predicted emotion for chunk {i*DURATION}-{(i+1)*DURATION}s: {predicted_emotion}')
        print(f'Confidence: {prediction[0][predicted_index[0]]*100:.2f}%\n')
        if i == 0:
            audio = AudioSegment.from_file(title)
            play(audio[:int(DURATION*1000)])
        else:
            # time.sleep(DURATION)
            audio = AudioSegment.from_file(title)
            play(audio[i*int(DURATION*1000):(i+1)*int(DURATION*1000)])
    print('\n')


def main():
    if len(sys.argv) != 2:
        print('Usage: python ser_demo.py <youtube_url>')
        sys.exit(1)

    print('\n\t' + '*' * 43)
    print('\t* Welcome to the emotion recognition demo *')
    print('\t' + '*' * 43 + '\n')

    title = download_audio(sys.argv[1]) + '.wav'
    predict_emotion(audio_to_features(title), title)


if __name__ == '__main__':
    main()
