<div align="center">
  <center><h1>Speech emotion recognition :microphone:</h1></center>
  </div>

  <div align="left">
  <left><h3>This project aims to classify emotions from speech. It is divided into two parts:</h3></left>
  </div>

-   Audio signal analysis through CNN/LSTM models
-   A script using a pre-trained transformer model with a speech to text library
<br>

<div align="center">
  <center><h1>Audio signal analysis through CNN/LSTM models :headphones:</h1></center>
  </div>

  <div align="center">
  <center><h2>Datasets and data pre-processing</h2></center>
  </div>

  <div align="left">
  <left><h3>For this project, we used three datasets:</h3></left>
  </div>

-   The [RAVDESS](https://zenodo.org/record/1188976#.YJZ6Zy1Q3OQ) dataset. It contains 1440 audio files (16-bit, 48kHz) with 60 different actors. Each actor recorded 2 different utterances for each of the 7 emotions (calm, happy, sad, angry, fearful, surprise, and disgust ).<br>

-   The [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) dataset. It contains 7442 audio files (16-bit, 48kHz) with 91 different actors. Each actor recorded 12 different utterances for each of the 6 emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad).<br>

-   The [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) dataset. It contains 2800 audio files (16-bit, 48kHz) with 2 different actors. Each actor recorded 7 different utterances for each of the 7 emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral).<br>

  <div align="left">
  <left><h3>Data pre-processing:</h3></left>
  </div>

We used the [librosa](https://librosa.org/) library to extract features from the audio files. The features extracted are:

-   Mel-frequency cepstral coefficients (MFCC)<br>
-   Chroma frequencies<br>
-   Mel-spectrogram<br>

<p align="center">
<img src="https://user-images.githubusercontent.com/91064070/235926461-bc649866-ee7f-4878-b292-7f6981c8d9c9.png"/>
</p>

The features are then split in train/validation sets and saved into .npy files.

The models were trained on different combinations of these datasets to find the best results. We found that merging the 3 datasets gave the most satisfying results, as it helped the model to generalize and avoid overfitting.

<div align="center">
  <center><h2>The models</h2></center>
  </div>

Several models were tried, replicating the ones we found in the literature. The best results were obtained with CNN models, followed by an hybrid CNN/LSTM model.

The models are composed of 5 to 8 convolutional layers, followed by Batch Normalization and max pooling at some points, according to the paper they were based on.<br>
The output is then flattened and fed to a dense layer. The output of the dense layer is then fed to a softmax layer with the number of emotions as output.<br>
The hybrid model adds 2 LSTM layers between the convolutional layers and the dense layer.<br>
You can check some of the models in the `/models/model_plot` folder. Two of them are saved and ready to be trained in the ```/model_training/train.py``` file.

We also made a custom class with [keras-tuner](https://keras.io/keras_tuner/) to find the best hyperparameters for the CNN model. You can find it in the `/model_training/CNNHypermodel.py` file. It is used in the ```/model_training/HM_train.py``` file.

<div align="center">
  <center><h2>Results</h2></center>
  </div>

Matrix confusion of the results of a CNN model trained on RAVDESS, excluding singing and calm emotions:
<p align="center">
<img src="https://user-images.githubusercontent.com/91064070/235926494-c18e0165-1f09-4049-888f-88b4ff37bba3.png"/>
</p>

<div align="center">
  <center><h2>How to use it</h2></center>
  </div>

UPDATE May 2023: The library used to download youtube videos seems to be deprecated. You can still download the audio files from youtube and convert them to wav with ffmpeg.

Play audio and print emotions by chunks of DURATION seconds:

```bash
$> python3 analyze_audio.py <audio.wav>
```

Download a youtube video and print emotions by chunks of DURATION seconds:

```bash
$> python3 ser_demo.py <youtube_url>
```

For a given model, print accuracy, precision, recall, f1-score for each class and the confusion matrix:

```bash
$> python3 stats.py <model>
```

<div align="left">
  <left><h3>Tools</h3></left>
  </div>

-   youtube_to_wav.py: download a youtube video and convert it to wav

-   data_extraction.py: extract features from dataset(s) and save x and y into .npy files

-   merge_datasets.py: concatenate two datasets by axis 0 and saves them into .npy files
<br>

<div align="center">
  <center><h1>Speech to text (transformers model) with 🤗</h1></center>
  </div>

Small script that uses existing libraries and a transformer model from [Hugging Face](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base).<br>
The script will:

1. Record audio from microphone
2. Transcribe audio to text
3. Translate to english if needed
4. Classify with a pre-trained transformer model

Use the following command to run the prediction from microphone:

```bash
$> python3 prediction_from_microphone.py
```

Then just speak when prompted. Text and emotion prediction ratios will be printed in the terminal.
