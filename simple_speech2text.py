import requests

# URL of the audio file to be downloaded
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hTqGqoC-LrW6S79HjuJUkg/trimmed-02.wav'

# Send a GET request to the URL to download the file
response = requests.get(url)

# Define the local file path where the audio file will be saved
audio_path_file = "sample-meeting.wav"

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # If successful, write the content to the specified local file path
    with open(audio_path_file, "wb") as file:
        file.write(response.content)
        print("File downloaded successfully")
else:
    print("Failed to download the file")

import torch
from transformers import pipeline

# Initialize the speech-to-text pipeline from HuggingFace Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
# The `chunk_length_s` parameter specifies the chunk length in seconds for processing
pipe = pipeline(
    "automatic-speech-recognition", # 指定任務類型是"語音辨識"
    model="openai/whisper-tiny.en", # 使用OpenAI的Whisper模型
    chunk_length_s=30 # 將音訊分為每段30s來處理，避免處理音訊太長時記憶體爆炸
)

# 常見的任務類型：
# - "text-classification"
# - "translation"
# - "summarization"
# - "question-answering"
# - "image-classification"
# - "automatic-speech-recognition"
# - "text-to-speech" (需要擴充)

# Define the path to the audio file needs to be transformed
sample = 'sample-meeting.wav'

# Perform speech recognition on the audio file 
# The `batch_size=8` parameter indicates how many chunks are processed at a time
# The result is stored in `prediction` with the key "text" containing the transcribed text
prediction = pipe(sample, batch_size=8)["text"] # 每次處理8個剛剛每30s為1 chunk的音訊
print(prediction)