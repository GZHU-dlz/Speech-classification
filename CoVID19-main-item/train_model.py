from random import random
from Function_model import ECAPA_TDNN
from data_clean import SoundDS, eval_SoundDS
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gc
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score, roc_auc_score, confusion_matrix

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 设置随机数种子
setup_seed(2010)
traindata_file = 'merge_all/merged_csv_part1234.csv'
evaldata_file = 'merge_all/merged_csv_part5.csv'
train_df = pd.read_csv(traindata_file)
train_df.head()
train_df['relative_path'] = train_df['path'].astype(str)
train_df = train_df[['relative_path', 'label']]
eval_df = pd.read_csv(evaldata_file)
eval_df.head()
eval_df['relative_path'] = eval_df['path'].astype(str)
eval_df = eval_df[['relative_path', 'label']]
train_ds = SoundDS(train_df)
eval_ds = eval_SoundDS(eval_df)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, num_workers=10, shuffle=True)
val_dl = torch.utils.data.DataLoader(eval_ds, batch_size=64, num_workers=10, shuffle=True)

myModel = ECAPA_TDNN(1024)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myModel = myModel.to(device)
next(myModel.parameters()).device

C = 1.0
lam = 0.2
sita = 0.3


# 训练
# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):

    def micro_f1(preds, truths):
        return f1_score(truths, preds, average='micro')

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    expdir = Path.cwd() / 'Covid19-model-save'
    version = 'mel_mfcc_ecapa1024'
    filename = 'only_celoss_sigmoid'
    logfilepath = expdir / version / filename / 'logs_{}.txt'.format(filename)
    logfilepath_eval = expdir / version / filename / 'logs_eval_{}.txt'.format(filename)
    model_savedir = expdir / version / filename

    def init_logger(self):
        self.logFileName = str(self.logfilepath)
        if self.logfilepath.exists == True:
            self.logfilepath.exists.unlink()
        if self.logfilepath_eval.exists == True:
            self.logfilepath_eval.exists.unlink()
        self.logFileName_eval = str(self.logfilepath_eval)

        with open(self.logFileName, 'a', encoding='utf-8') as wf:
            wf.write("*" * 100 + "/n")
            wf.close()

    def write_line2log(log_dict: dict, filedir, isprint: True):
        strp = ''
        with open(filedir, 'a', encoding='utf-8') as f:
            for key, value in log_dict.items():
                witem = '{}'.format(key) + ':{},'.format(value)
                strp += witem
            f.write(strp)
            f.write('/n')
        if isprint:
            print(strp)
        pass

    def print_network(model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("Model {},the number of parameters: {}".format(name, num_params))

    def save_model(i):
        pdict = {"model": model.state_dict(),
                 }
        path = model_savedir / "{:04}.pth".format(i)
        torch.save(pdict, str(path))
        print("---------------- model saved ------------------- ")

    print_network(myModel, 'ecapa_1024')

    def test_acc(model, val_dl):
        correct_prediction = 0
        total_prediction = 0

        truths = []
        preds = []
        all_output = []
        with torch.no_grad():
            for data in val_dl:
                inputs, labels,inputs_mel = data[0].to(device), data[1].to(device),data[2].to(device)
                inputs = torch.squeeze(inputs)
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                inputs_mel = torch.squeeze(inputs_mel)
                inputs_mm, inputs_ss = inputs_mel.mean(), inputs_mel.std()
                inputs_mel = (inputs_mel - inputs_mm) / inputs_ss


                outputs = torch.sigmoid(model(inputs,inputs_mel)[0])
                prediction = outputs.detach()

                prediction = prediction.cpu()
                prediction = np.argmax(prediction, axis=1)
                labels = labels.cpu()
                truths.append(labels)
                preds.append(prediction)
                outputs = outputs.cpu().numpy()
                all_output.append(outputs)

        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        all_output = np.concatenate(all_output)
        acc1 = accuracy_score(truths, preds)
        con_matrix = confusion_matrix(truths, preds)
        all_output=all_output[:,1]
        auc=roc_auc_score(truths, all_output)


        losse_curves = {"eval_step--": "",
                        "Epoch": epoch,
                        "Accuracy": acc1,
                        "Auc":auc,
                        }
        write_line2log(losse_curves, logfilepath_eval, isprint=True)

    for epoch in range(1, num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        truths = []
        preds = []

        for i, data in enumerate(train_dl):
            inputs, labels,inputs_mel = data[0].to(device), data[1].to(device),data[2].to(device)
            inputs = torch.squeeze(inputs)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()  # 标准差
            inputs = (inputs - inputs_m) / inputs_s
            inputs_mel=torch.squeeze(inputs_mel)
            inputs_mean,inputs_std=inputs_mel.mean(),inputs_mel.std()
            inputs_mel=(inputs_mel - inputs_mean)/inputs_std

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs,inputs_mel)[0])
            fc8=model(inputs,inputs_mel)[1]
            num_pos=torch.sum(labels)
            r_loss=torch.mean(torch.square(fc8.weight.data))
            pre_outputs=torch.max(outputs,dim=1)
            hinge_loss = torch.sum(torch.maximum(torch.tensor((0.)), labels * (1. - pre_outputs.values * labels))) / num_pos
            labels_one_hot = F.one_hot(labels, num_classes=2)
            labels_one_hot = labels_one_hot.float()
            outputs=outputs.float()

            bce_loss = criterion(outputs, labels_one_hot)
            ce_loss=(1-lam)*bce_loss+lam*C*hinge_loss +sita*r_loss

            ce_loss.backward()
            optimizer.step()
            scheduler.step()

            prediction = outputs.detach()
            prediction=prediction.cpu()
            prediction = np.argmax(prediction, axis=1)
            labels = labels.cpu()
            truths.append(labels)
            preds.append(prediction)
            running_loss += ce_loss.item()
            gc.collect()
            torch.cuda.empty_cache()

        num_batches = len(train_dl)
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        acc = accuracy_score(truths, preds)
        avg_loss = running_loss / num_batches
        losse_curves = {"Epoch": epoch,
                        "Loss": avg_loss,
                        "acc": acc
                        }
        if epoch == 1:
            print("create loss dict")
            loss_log_dict = {}
            for k, v in losse_curves.items():
                loss_log_dict[k] = []
            print("loss dict created")
        for k, v in loss_log_dict.items():
            loss_log_dict[k].append(losse_curves[k])  # 把每batch的loss数据加入到 loss curves中
        write_line2log(losse_curves, logfilepath, isprint=True)

        if epoch % 20 == 0:
            test_acc(myModel, val_dl)
            save_model(epoch)
        gc.collect()
        torch.cuda.empty_cache()

    print('Finished Training')

if __name__=="__main__":
    num_epochs = 81
    training(myModel, train_dl, num_epochs)