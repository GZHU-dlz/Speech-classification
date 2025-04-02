from scipy.ndimage import binary_dilation
import librosa
import numpy as np
import struct
import librosa.display
import webrtcvad
import soundfile as sf

class vadect(object):
    def __init__(self, audio_file, i):
        self.audio_file = audio_file
        print(f'path:{self.audio_file}')
        self.i = i

    def moving_average(array, width):
          array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
          ret = np.cumsum(array_padded, dtype=float)
          ret[width:] = ret[width:] - ret[:-width]
          return ret[width - 1:] / width
    def vade(self):
        int16_max = (2 ** 15) - 1
        wav, source_sr = librosa.load(self.audio_file, sr=16000)
        print(f'sr:{source_sr}')
        samples_per_window = source_sr // 10
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
        voice_flags = []
        vad = webrtcvad.Vad(mode=self.i)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                     sample_rate=16000))
            voice_flags = np.array(voice_flags)
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width
        audio_mask = moving_average(voice_flags, 8)
        audio_mask = np.round(audio_mask).astype(np.bool)
        audio_mask = binary_dilation(audio_mask, np.ones(6 + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        res=wav[audio_mask == True]
        print(f'vad_i:{self.i}')
        sf.write("/private/covid/test/0.wav".format(self.i), res.astype(np.float32), 16000, subtype='PCM_24')

if __name__=="__main__":
    audio_path = '/private/covid/test/447_female_cough.wav'
    vad_mode = 1
    print('12123')
    vad_class = vadect(audio_path, vad_mode)
    vad_class.vade()
