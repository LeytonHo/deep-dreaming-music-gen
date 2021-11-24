import librosa
import os
import numpy as np
import warnings

def preprocess(rootdir):
    """
    Preprocess mp3 data by converting to numpy array using the librosa library.
    Inputs:
    - rootdir: Directory housing input mp3 files
    Returns:
    - audio_data: n
    """
    audio_data = []
    sr_data = []
    genre_data = [] #np.array([])

    for subdir, _, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.mp3'):
                try:
                    filepath = subdir + '/' + file
                    audio, sr = librosa.load(filepath)
                    audio_data += [audio]
                    sr_data += [sr]

                except RuntimeError:
                    # Ignore malformed mp3 files
                    continue

    return audio_data, sr_data

# Suppress expected warnings from librosa, source: https://github.com/librosa/librosa/issues/1015#issuecomment-552506963
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    preprocess('fma_small')
