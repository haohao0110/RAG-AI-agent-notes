import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length=30
    )

    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Set up Gradio Interface
audio_input = gr.Audio(sources="upload", type="filepath") # Audio input
output_text = gr.Textbox() # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription App",
    description="Upload the audio file"
)

# Launch the Gradio app
iface.launch(server_name = "0.0.0.0", server_port=5000)