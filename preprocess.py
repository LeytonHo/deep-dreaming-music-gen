import librosa
import os
import numpy as np
import warnings
import utils

def preprocess(rootdir):
    """
    Preprocess mp3 data by converting to numpy array using the librosa library.
    Inputs:
    - rootdir: Directory housing input mp3 files
    Returns:
    - audio_data: Vectorized audio data as amplitude of waveform at each t in time series
    - sr_data: Sampling rate of each track
    - genre_data: Genre of each track
    """
    truncated_size = 660984 # Sizes of audio files are 660984 and 661560 (time series points). Truncating to 660984 for simplicity.
    audio_data = np.empty((truncated_size,0))
    sr_data = np.array([])
    genre_data = np.array([])

    # Reference: https://notebook.community/mdeff/fma/usage
    tracks = utils.load('fma_metadata/tracks.csv')
    small = tracks[tracks['set', 'subset'] <= 'small']
    track_id_to_genre = small['track', 'genres'].to_dict()

    for subdir, _, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.mp3'):
                try:
                    filepath = subdir + '/' + file
                    track_id = int(os.path.splitext(file)[0])
                    genre = track_id_to_genre[track_id]

                    # Don't process multi-genre tracks
                    if len(genre) > 1:
                        continue

                    audio, sr = librosa.load(filepath)
                    audio_data = np.append(audio_data, np.reshape(audio[:truncated_size], (truncated_size, 1)), axis=1)
                    sr_data = np.append(sr_data, sr)
                    genre_data = np.append(genre_data, genre)

                except RuntimeError:
                    # Ignore malformed mp3 files
                    continue

    return audio_data, sr_data, genre_data

# Suppress expected warnings from librosa, source: https://github.com/librosa/librosa/issues/1015#issuecomment-552506963
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    preprocess('fma_small')
