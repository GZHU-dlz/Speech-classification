from pathlib import Path  ###  某种程度上来说这个库确实比 os好用一些。
import torch
import pickle
class Create_Train_Hparams():
    def __init__(self):
        self.n_mels = 80
        self.use_meldatadir_name = './meldata_22k_trimed'
        self.total_iters = 1000
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_every = 200
        self.lr_update_every = 200
        self.eval_every = 50
        self.train_ratio  = 0.9
        self.mel_seglen = 256
        self.min_train_mellen = 120
        self.batchsize_train  = 12
        self.lr_start = 0.0005
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.amsgrad = True
        self.weight_decay  = 0.0001
        self.grad_norm  = 3
        self.is_lr_decay = False
        self.speaker_nums = 5
        self.epdir = Path('./Experiments')
        self.ep_version = None
        self.ep_version_dir = None
        self.model_savedir = None
        ## files create
        self.hp_filepath = None
        self.ep_logfilepath = None
        self.ep_logfilepath_eval = None

        pass
    def set_experiment(self,version='v1'):
        self.ep_version = version
        self.ep_version_dir = self.epdir / self.ep_version
        self.model_savedir = self.ep_version_dir / 'checkpoints_{}'.format(version)
        self.conversion_dir = self.ep_version_dir / 'conversion_result_{}'.format(version)
        self.hp_filepath = self.ep_version_dir.joinpath('hparams_{}.pickle'.format(version))
        self.ep_logfilepath = self.ep_version_dir / 'logs_{}.txt'.format(version)
        self.ep_logfilepath_eval = self.ep_version_dir / 'logs_eval_{}.txt'.format(version)

class Create_Prepro_Hparams():
    def __init__(self):
        self.wav_datadir_name = 'speaker_verify_dataset'
        self.feature_dir_name = 'meldata_22k_trimed'
        self.trim_db = 20
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.sample_rate = 48000
        self.f_min = 0
        self.f_max = 24000
        self.n_mels = 80

    def set_preprocess_dir(self, wav_datadir, feature_dir):
        self.wav_datadir_name = wav_datadir
        self.feature_dir_name = feature_dir
        pass
###  新的参数文件的生成.
def boot_a_new_experiment_hp(epversion,tt_iters=None,mel_segm_len=None,train_batchsize=None,use_mel_dir=None,
                             start_lr=None):
    hp = Create_Train_Hparams()
    hp.set_experiment(epversion)
    if tt_iters != None:
        hp.total_iters = tt_iters
    if mel_segm_len != None:
        hp.mel_seglen = mel_segm_len
    if train_batchsize != None:
        hp.batchsize_train = train_batchsize
    if use_mel_dir != None:
        hp.use_meldatadir = use_mel_dir
    if start_lr != None:
        hp.lr_start = start_lr
    return hp

def boot_a_new_experiment(epversion, tt_iters = None, mel_segm_len = None, train_batchsize = None, use_mel_dir = None,
    start_lr = None):
    hp = boot_a_new_experiment_hp(epversion, tt_iters, mel_segm_len , train_batchsize , use_mel_dir ,
    start_lr)
    hp.ep_version_dir.mkdir(parents=True, exist_ok=True) ## Experiment/v1/
    hp.model_savedir.mkdir(parents=True, exist_ok=True)  ## Experiment/v1/checkpoints
    with open(hp.hp_filepath.resolve(),'wb') as hpf:
        pickle.dump(hp,hpf)
    return hp
    pass
if __name__ == "__main__":
    pass











