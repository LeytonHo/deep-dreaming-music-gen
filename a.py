import librosa
import soundfile as sf
from mutagen.mp3 import MP3  
from mutagen.easyid3 import EasyID3  
import mutagen.id3  
from mutagen.id3 import ID3, TIT2, TIT3, TALB, TPE1, TRCK, TYER 
import glob 

audio_data = 'fma_small/000/000002.mp3'
x , sr = librosa.load(audio_data, 11025)
print(x.shape)#<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 22050


filez = glob.glob("fma_small/000/000002.mp3")  
mp3file = MP3(filez[0], ID3=EasyID3)  
print(mp3file['genre'])

# sf.write('stereo_file1.wav', x, sr)