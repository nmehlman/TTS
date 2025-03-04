import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models().list_tts_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
tts.tts_to_file(text="Hello world!", speaker_wav="/home1/nmehlman/nick_codebase/misc/test_audio.wav", language="en", file_path="output.wav")