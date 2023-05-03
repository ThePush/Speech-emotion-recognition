import wave
import pyaudio
from googletrans import Translator
from transformers import pipeline
import speech_recognition as sr
import nltk
nltk.download('omw-1.4')

classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base", top_k=None)


r = sr.Recognizer()
translator = Translator()

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 4
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording audio...')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Recording complete')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()


with sr.AudioFile('output.wav') as source:
    audio_data = r.record(source)
    print('Speech recognition...')
    text = r.recognize_google(audio_data, language='fr-FR')
    print(text)
    translated = translator.translate(text, dest='en')
    print(translated.text)
    print('Sentiment classification...')
    print(classifier(translated.text))
