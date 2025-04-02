from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset
class SoundDS(Dataset): 
  def __init__(self, df): 
    self.df = df

  def __len__(self): 
    return len(self.df)     

  def __getitem__(self, idx):
    audio_file = self.df.loc[idx, 'relative_path']
    class_id = self.df.loc[idx, 'label']
    print(class_id)
   
    return audio_file, class_id
  
traindata_file = '/private/Coswara-Data/cut_5s_data/cough_quality12_5s_train0.8_data.csv'
evaldata_file = '/private/Coswara-Data/cut_5s_data/cough_quality12_5s_test0.2_data.csv'

skfolds = StratifiedKFold(n_splits=3, random_state=25, shuffle=True)
