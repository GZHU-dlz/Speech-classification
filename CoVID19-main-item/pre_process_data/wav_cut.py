'''
切割音频的脚本
'''

import os
import wave
import numpy as np

audio_class = 'vowel_e'
CutTimeDef = 5

filename='/private/Coswara-Data/positive_quality12_id_scp/positive_quality12_{}_new_path.scp'.format(audio_class)
file = open(filename)
files = [ f for f in file.read().splitlines() ]

wav_lt_cutTime = []
def SetFileName(WavFileName):
    for i in range(len(files)):
        FileName = files[i]
        print("SetFileName File Name is ", FileName)
        FileName = WavFileName

def CutFile():
    for i in range(len(files)):
        FileName = files[i]
        print("CutFile File Name is ", FileName)
        f = wave.open(r"" + FileName, "rb")
        params = f.getparams()
        print(params)
        nchannels, sampwidth, framerate, nframes = params[:4]
        CutFrameNum = framerate * CutTimeDef
        print("CutFrameNum=%d" % (CutFrameNum))
        print("nchannels=%d" % (nchannels))
        print("sampwidth=%d" % (sampwidth))
        print("framerate=%d" % (framerate))
        print("nframes=%d" % (nframes))
        str_data = f.readframes(nframes)
        f.close
        wave_data = np.fromstring(str_data, dtype=np.short)
        print(wave_data.shape)
        wave_data = wave_data.T
        temp_data = wave_data.T
        print(temp_data.shape)
        StepNum = int(CutFrameNum)
        StepTotalNum = StepNum;
        haha = 0
        while StepTotalNum <= nframes:
            print("stemp=%d" % (haha))
            print(files[i])
            fn = os.path.basename(files[i])
            print(StepNum)
            FileName = '/private/Coswara-Data/cut_5s_data/positive_data/{}/'.format(audio_class) + fn.split(".")[0] +"-"+ str(haha+1) + ".wav"
            print(FileName)
            temp_dataTemp = temp_data[StepNum * (haha):StepNum * (haha+1)]
            haha = haha + 1;
            StepTotalNum = haha * StepNum + StepNum;
            temp_dataTemp = temp_dataTemp.astype(np.short) # 打开wav文档
            f = wave.open(FileName, "wb")
            f.setnchannels(nchannels)
            f.setsampwidth(sampwidth)
            f.setframerate(framerate)
            f.writeframes(temp_dataTemp.tostring())
            f.close()
        if StepNum > nframes:
            wav_lt_cutTime.append(FileName)
    with open('/private/Coswara-Data/cut_5s_data/lt_5s_data_path/positive_{}_lt_5s.scp'.format(audio_class),'w') as p:
        for path in wav_lt_cutTime:
            p.write(path + '\n')

if __name__ == '__main__' :
    CutFile()
    print("Run Over!")
