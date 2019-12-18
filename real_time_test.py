from spleeter.separator import Separator
import glob
import tqdm
import os
import shutil
import librosa
import time
import numpy as np
# Using embedded configuration.

if __name__ == "__main__":
        
    separator = Separator('spleeter:2stems')

    # # Use audio loader explicitly for loading audio waveform :
    # from spleeter.audio.adapter import get_default_audio_adapter

    # audio_loader = get_default_audio_adapter()
    # sample_rate = 44100
    # waveform, _ = audio_loader.load('/path/to/audio/file', sample_rate=sample_rate)

    # Perform the separation :

    # for f in tqdm.tqdm(glob.glob("./seperate/*.wav")):
    f = "audio_example.mp3"
    y,sr = librosa.load(librosa.util.example_audio_file(),sr=None,mono=False)



    yx = np.hstack([y])
    duration = yx.shape[1]/sr
    print(duration)
    time.sleep(4)
    start_time = time.time()

    silce_t = duration/10
    prediction = separator.separate(yx.T)
    for i in tqdm.tqdm(range(10)):
        start = int(sr*silce_t*i)
        end = int(sr*silce_t*(i+1))
        
        silce = yx[:,start:end]
        prediction = separator.separate(silce.T)
        time.sleep(4)

    print(duration,time.time()-start_time)
