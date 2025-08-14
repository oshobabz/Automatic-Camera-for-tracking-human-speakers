# --- File: vad.py ---
import torch
import torchaudio
import pyaudio
import numpy as np

class VAD:
    def __init__(self, sample_rate=16000, chunk_size=512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.model, _ = torch.hub.load('snakers4/silero-vad', model='silero_vad')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size)

    def is_speech(self, data):
        audio = np.frombuffer(data, dtype=np.int16).copy()
        audio_tensor = torch.from_numpy(audio).float() / 32768.0
        audio_tensor = audio_tensor.unsqueeze(0)
        confidence = self.model(audio_tensor, self.sample_rate).item()
        return confidence > 0.5

    def detect_speech(self, duration=5):
        frames = int(self.sample_rate * duration / self.chunk_size)
        for _ in range(frames):
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            if self.is_speech(data):
                return True
        return False

    def read(self):
        return self.stream.read(self.chunk_size, exception_on_overflow=False)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()