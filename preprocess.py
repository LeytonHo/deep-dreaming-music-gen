import librosa
import os
import numpy as np
import warnings
import utils
import pickle

# Sizes of audio files are 330492 and 330780 (time series points). Truncating to 330492 for simplicity.
INPUT_SIZE = 55125
SAMPLE_RATE = 11025

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
    truncated_size = INPUT_SIZE 
    duration = 5 # seconds
    audio_data = np.empty((0,truncated_size))
    sr_data = np.array([])
    genre_data = np.array([])

    # Reference: https://notebook.community/mdeff/fma/usage
    tracks = utils.load('fma_metadata/tracks.csv')
    small = tracks[tracks['set', 'subset'] <= 'small']
    track_id_to_genre = small['track', 'genres'].to_dict()

    for subdir, _, files in os.walk(rootdir):
        for file in files:
            #if file.endswith('.mp3') and np.shape(sr_data)[0] < 100:
            if file.endswith('.mp3'):
                try:
                    filepath = subdir + '/' + file
                    track_id = int(os.path.splitext(file)[0])
                    genre = track_id_to_genre[track_id]

                    # Don't process multi-genre tracks
                    if len(genre) > 1:
                        continue

                    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=duration)
                    audio_data = np.append(audio_data, np.reshape(audio[:truncated_size], (1, truncated_size)), axis=0)
                    sr_data = np.append(sr_data, sr)
                    genre_data = np.append(genre_data, genre)

                    if np.shape(sr_data)[0] % 100 == 0:
                        print("finished ", np.shape(sr_data)[0])

                except RuntimeError:
                    # Ignore malformed mp3 files
                    continue
            
                except Exception:
                    # Ignore malformed mp3 files
                    continue

    with open('preprocessed.pickle', 'wb') as f:
        pickle.dump((audio_data, sr_data, genre_data), f)

    return audio_data, sr_data, genre_data

# Suppress expected warnings from librosa, source: https://github.com/librosa/librosa/issues/1015#issuecomment-552506963
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    preprocess('fma_small')
