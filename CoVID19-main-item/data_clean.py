from tkinter.tix import InputOnly

import torchaudio
from torchaudio import transforms
import torch
import random
from torch.utils.data import  Dataset
import numpy as np
class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    # audio_file = download_path/'fold1'/'101415-3-0-2.wav'
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file, normalize=True)
        return (sig, sr)

    # 标准化采样率
    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    # 数据扩充增广（时移）
    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        #     print(f"afterpad_sig_shape:{sig.shape}")#[2, 220000]
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # 梅尔谱图
    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        # print(f'sig_shape:{sig.shape}')
        #     print(f"after_timeshift_sig_shape:{sig.shape}")#[2, 220000]维度不变
        top_db = 80
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        mel_spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        mel_spec = transforms.AmplitudeToDB(top_db=top_db)(mel_spec)
        return (mel_spec)

    @staticmethod
    def spec_mfcc(aud, n_mfcc=40, n_fft=400, hop_len=200, n_mels=80):
        sig, sr = aud
        top_db = 80
        mfcc = transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                               melkwargs={"n_fft": n_fft, "hop_length": hop_len, "n_mels": n_mels})(sig)
        transforms.AmplitudeToDB(top_db=top_db)(mfcc)

        return (mfcc)

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

    @staticmethod
    def mfcc_augment(mfcc, max_mask_time=10, max_mask_freq=5):
        # 时间掩码
        _,n_frames, n_coeffs = mfcc.shape
        if max_mask_time > 0:
            for _ in range(2):  # 掩盖两次
                t = np.random.randint(0, max_mask_time)
                start = np.random.randint(0, n_frames - t)
                mfcc[start:start + t, :] = 0  # 掩盖时间轴

        # 频率掩码（调整为Mel系数维度）
        if max_mask_freq > 0:
            for _ in range(2):
                f = np.random.randint(0, max_mask_freq)
                start = np.random.randint(0, n_coeffs - f)
                mfcc[:, start:start + f] = 0  # 掩盖频率轴

        return mfcc


# 自定义数据加载器
# from torch.utils.data import DataLoader, Dataset, random_split
# import torchaudio

# ----------------------------
# Sound Dataset
# ----------------------------
# SoundDS：MyDataset类


class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df  # df为csv文件
        self.duration = 5000  # 固定长度，单位为ms
        self.sr = 16000  # 采样率
        self.channel = 1  # 通道数
        self.shift_pct = 0.4
        self.lowcut = 500

        # ----------------------------

    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.df.loc[
            idx, 'relative_path']  # .loc:取idx对应行的所有数据，.loc[idx, 'relative_path']:取idx对应行的relative_path
        # print(audio_file)

        # Get the Class ID
        class_id = self.df.loc[idx, 'label']  # 取idx对应行的classID
        # print(class_id)
        aud = AudioUtil.open(audio_file)
        #     AudioUtil.show_wave(aud)
        # 有些声音有更高的采样率，或者比大多数声音更少的通道。所以让所有声音都有相同数量的通道和相同的采样率。除非采样速率相同，否则pad_trunc仍然会产生不同长度的数组，即使声音持续时间相同。
        reaud = AudioUtil.resample(aud, self.sr)  # 标准化采样率
        shift_aud = AudioUtil.time_shift(reaud, self.shift_pct)  # 时移
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=80, n_fft=400, hop_len=200)  # 转换为mel谱图
        mfcc = AudioUtil.spec_mfcc(shift_aud, n_mfcc=40) #提取mfcc
        # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2) #时间和频率屏蔽
        aug_mfcc = AudioUtil.spectro_augment(sgram, max_mask_pct=0.2, n_freq_masks=2, n_time_masks=2)
        mfcc=AudioUtil.mfcc_augment(mfcc, max_mask_time=10, max_mask_freq=5)
        return aug_mfcc, class_id,mfcc


class eval_SoundDS(Dataset):
    def __init__(self, df):
        self.df = df  # df为csv文件
        self.duration = 5000  # 固定长度，单位为ms
        self.sr = 16000  # 采样率
        self.shift_pct = 0.4

        # ----------------------------

    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.df.loc[
            idx, 'relative_path']  # .loc:取idx对应行的所有数据，.loc[idx, 'relative_path']:取idx对应行的relative_path
        # Get the Class ID
        class_id = self.df.loc[idx, 'label']  # 取idx对应行的classID
        aud = AudioUtil.open(audio_file)
        # 有些声音有更高的采样率，或者比大多数声音更少的通道。所以让所有声音都有相同数量的通道和相同的采样率。除非采样速率相同，否则pad_trunc仍然会产生不同长度的数组，即使声音持续时间相同。
        reaud = AudioUtil.resample(aud, self.sr)  # 标准化采样率
        sgram = AudioUtil.spectro_gram(reaud, n_mels=80, n_fft=400, hop_len=200)  # 转换为mel谱图
        mfcc = AudioUtil.spec_mfcc(reaud, n_mfcc=40) #提取mfcc

        return sgram, class_id,mfcc