import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import os

def transcribe_audio(segment):
    recognizer = sr.Recognizer()
    with sr.AudioFile(segment) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "and"
        except sr.RequestError as e:
            return f"and"

def sentiment_analysis(audio_file_path):

  extension = audio_file_path.split(".")
  extension = extension[-1]

  audio = AudioSegment.from_file(audio_file_path, format=extension)
  segments = split_on_silence(audio, silence_thresh=-40)

  transcriptions=[]
  full_transcript=""
  for i, segment in enumerate(segments):
      segment.export(f"segment_{i + 1}.wav", format="wav")
      transcription = transcribe_audio(f"segment_{i + 1}.wav")
      transcriptions.append(transcription)
      full_transcript=full_transcript+" "+transcription
      os.remove(f"segment_{i + 1}.wav")

  sentiment_scores = []
  for text in transcriptions:
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
    sentiment_scores.append(sentiment_score)

  compound_scores = [score['compound'] for score in sentiment_scores]

  return full_transcript,transcriptions,compound_scores

# sentiment_analysis("hotel_room.mp3")