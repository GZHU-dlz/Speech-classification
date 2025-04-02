import torch
import librosa
import numpy as  np
import librosa.util as librosa_util
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from Create_Hparams import Create_Prepro_Hparams,Create_Train_Hparams
from pathlib import Path
from matplotlib import pyplot as plt

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        self.num_samples = num_samples
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x
def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

## 对谱的反动态压缩
def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=11025):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        y = y.clamp(min=-1, max=1)
        # assert(torch.min(y.data) >= -1)
        # assert(torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

def extract_mel_feature_bytaco(hp:Create_Prepro_Hparams):
    '''
    :param hp:
    将数据集的语音 保持原始目录结构提取到 另一个文件夹
    :return:
    '''
    stftfunc = TacotronSTFT(filter_length=hp.n_fft,
                            hop_length=hp.hop_length,
                            win_length=hp.win_length,
                            n_mel_channels=hp.n_mels,
                            sampling_rate=hp.sample_rate,
                            mel_fmin=hp.f_min,
                            mel_fmax=hp.f_max)

    print("使用Taco2 mel 提取")

    '''
    # create the file struct in the new place, except the .wav
    for dirname, subdir, files in os.walk(datadir):# os.walk是获取所有的目录
            if dirname == datadir:
                pass
            else:
                dn = dirname.replace(datadir, out_dir)
                os.mkdir(dn)
    
    '''
    src_wavp = Path(hp.wav_datadir_name)
    for x in src_wavp.rglob('*'):
        if x.is_dir():
            Path(str(x.resolve()).replace(hp.wav_datadir_name,hp.feature_dir_name)).mkdir(parents=True,exist_ok=True)
    print("*" * 20)
    wavpaths = [ x  for x in src_wavp.rglob('*.wav') if x.is_file() ]
    ttsum = len(wavpaths)
    mel_frames = []
    k = 0
    for wp in wavpaths:
        k += 1
        the_wavpath = str(wp.resolve())
        the_melpath = str(wp.resolve()).replace(hp.wav_datadir_name,hp.feature_dir_name).replace('wav','npy')
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=20)
        wavform = torch.FloatTensor(wavform).unsqueeze(0)
        mel = stftfunc.mel_spectrogram(wavform)
        mel = mel.squeeze().detach().cpu().numpy()
        np.save(the_melpath, mel)
        mel_frames.append(mel.shape[-1])
        print("{}|{} -- mel_length:{}".format(k,ttsum,mel.shape[-1]))

    mean_len = sum(mel_frames) / len(mel_frames)
    max_fl = max(mel_frames)
    min_fl = min(mel_frames)
    print("*" * 100)
    print("Melspec length , Mean:{},Max:{},min:{}".format(mean_len, max_fl, min_fl))

    pass

def plot_hist_of_meldata(datadirname):
    datadirp = Path(datadirname)

    mellens = []
    wavpaths = [ x for x in datadirp.rglob('*.npy') if x.is_file() ]

    for wavp in wavpaths:
        mel = np.load(str(wavp))
        mellens.append(mel.shape[-1])

    max_ = max(mellens)
    min_ = min(mellens)
    avg_ = int(sum(mellens)/len(mellens))
    print("max:{},min:{},avg:{}".format(max_,min_,avg_) )

    plt.figure()
    plt.title("MelLens_hist_" + "max:{},min:{},avg:{}".format(max_,min_,avg_))
    plt.hist(mellens)
    plt.xlabel("mel length")
    plt.ylabel("numbers")
    plt.savefig("Mel_lengths_hist" )
    plt.show()

if __name__=="__main__":
    wav_datadir_name = 'breathing'
    feature_dir_name = 'mel_spec_breathing'
    preprocess_hp = Create_Prepro_Hparams()
    preprocess_hp.set_preprocess_dir(wav_datadir_name,feature_dir_name) # 设置 源数据路径和目标路径
    extract_mel_feature_bytaco(preprocess_hp)
    plot_hist_of_meldata("meldata_22k_trimed")
