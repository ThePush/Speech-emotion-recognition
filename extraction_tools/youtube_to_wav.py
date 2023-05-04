import pytube
import io
from pydub import AudioSegment

# Set the URL of the YouTube video you want to extract audio from
url = 'https://www.youtube.com/watch?v=jNQXAC9IVRw'

# Extract the audio stream from the YouTube video using pytube
youtube = pytube.YouTube(url)
audio_stream = youtube.streams.filter(only_audio=True).first()

# Convert the audio stream to a byte string using pytube
audio_data = io.BytesIO()
audio_stream.stream_to_buffer(audio_data)
audio_data.seek(0)

# Convert the byte string to an AudioSegment using pydub
audio_segment = AudioSegment.from_file(audio_data)

# Save the audio segment as a .wav file
audio_segment.export("output.wav", format="wav")
