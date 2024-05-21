import torch
import numpy as np
from scipy.io import wavfile
import torchaudio
from matplotlib import pyplot as plt

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def generate_spectrograms(audio):
    
    return audio

