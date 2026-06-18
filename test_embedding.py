import librosa
import torch
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained(
    "anhnct/qwen3-voice-embedding-12hz-1.7B-batch", trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    "anhnct/qwen3-voice-embedding-12hz-1.7B-batch", trust_remote_code=True,
)

# processor.push_to_hub("anhnct/qwen3-voice-embedding-12hz-1.7B-batch")
# model.push_to_hub("anhnct/qwen3-voice-embedding-12hz-1.7B-batch")
model.eval()

audio, sr = librosa.load("/home/anhnct/project/Dragon-TTS/normal.wav", sr=None, mono=True)

audio_1, sr = librosa.load("/home/anhnct/project/Dragon-TTS/sample0.wav", sr=None, mono=True)

inputs = processor([audio, audio_1], sampling_rate=sr)
print(inputs)

with torch.no_grad():
    embedding = model(**inputs, return_features=True)  # (1, 2048)
print(embedding.keys())

#print(embedding.shape)
