import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import pickle
from model import NextMedPredModel, RefinePrediction
from utils import compute_accuracy, compute_auc
from collections import Counter
import random

class EHRPredDataset(Dataset):
    def __init__(self, data):
        with open(data, 'rb') as file:
            self.data = pickle.load(file)
        self.max_len = max(len(ts) for ts in self.data) - 1  

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pad_sequences(sequences, max_len, dtype='float32'):
        #if not max_len:
        #    max_len = max(len(seq) for seq in sequences)
        
        padded_seqs = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padded_seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]), dtype=dtype)])
            else:
                padded_seq = seq
            padded_seqs.append(padded_seq)
        
        return np.stack(padded_seqs)

    def __getitem__(self, idx):
        ts = self.data[idx]
        seq_len = len(ts) - 1

        vital_indices = list(range(10, ts.shape[1] - 3))
        vitals = ts.iloc[:-1, vital_indices].values
        pre_meds = ts.iloc[:-1, 6:10].values
        demo_info = ts.iloc[0, 2:6].values
        note_info = np.array(ts.iloc[:-1, -1].tolist())
        
        vitals = self.pad_sequences([vitals], self.max_len)[0]
        pre_meds = self.pad_sequences([pre_meds], self.max_len)[0]
        note_info = self.pad_sequences([note_info], self.max_len)[0]

        label = ts.iloc[-1, -3]

        vitals = torch.tensor(vitals, dtype=torch.float)
        pre_meds = torch.tensor(pre_meds, dtype=torch.float)
        note_info = torch.tensor(note_info, dtype=torch.float).squeeze()
        label = torch.tensor(label, dtype=torch.float)
        seq_len = torch.tensor(seq_len, dtype=torch.float)
        demo_info = torch.tensor(demo_info.astype(np.float64), dtype=torch.float)

        return vitals, pre_meds, note_info, demo_info, seq_len, label

class ReliableEHRPredDataset(Dataset):
    def __init__(self, data):
        with open(data, 'rb') as file:
            raw_data = pickle.load(file)
        self.data = [ts for ts in raw_data if ts.iloc[-1, -2] == 0.0]
        self.max_len = max(len(ts) for ts in self.data) - 1 

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pad_sequences(sequences, max_len, dtype='float32'):
        padded_seqs = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padded_seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]), dtype=dtype)])
            else:
                padded_seq = seq
            padded_seqs.append(padded_seq)
        
        return np.stack(padded_seqs)

    def __getitem__(self, idx):
        ts = self.data[idx]
        seq_len = len(ts) - 1

        vital_indices = list(range(10, ts.shape[1] - 3))
        vitals = ts.iloc[:-1, vital_indices].values
        pre_meds = ts.iloc[:-1, 6:10].values
        demo_info = ts.iloc[0, 2:6].values
        note_info = np.array(ts.iloc[:-1, -1].tolist())
        
        vitals = self.pad_sequences([vitals], self.max_len)[0]
        pre_meds = self.pad_sequences([pre_meds], self.max_len)[0]
        note_info = self.pad_sequences([note_info], self.max_len)[0]

        label = ts.iloc[-1, -3]

        vitals = torch.tensor(vitals, dtype=torch.float)
        pre_meds = torch.tensor(pre_meds, dtype=torch.float)
        note_info = torch.tensor(note_info, dtype=torch.float).squeeze()
        label = torch.tensor(label, dtype=torch.float)
        seq_len = torch.tensor(seq_len, dtype=torch.float)
        demo_info = torch.tensor(demo_info.astype(np.float64), dtype=torch.float)

        return vitals, pre_meds, note_info, demo_info, seq_len, label

class UncerntainEHRPredDataset(Dataset):
    def __init__(self, data):
        with open(data, 'rb') as file:
            raw_data = pickle.load(file)
        self.data = [ts for ts in raw_data if ts.iloc[-1, -2] == 1.0]
        self.max_len = max(len(ts) for ts in self.data) - 1 

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pad_sequences(sequences, max_len, dtype='float32'):
        padded_seqs = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padded_seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]), dtype=dtype)])
            else:
                padded_seq = seq
            padded_seqs.append(padded_seq)
        
        return np.stack(padded_seqs)

    def __getitem__(self, idx):
        ts = self.data[idx]
        seq_len = len(ts) - 1

        vital_indices = list(range(10, ts.shape[1] - 3))
        vitals = ts.iloc[:-1, vital_indices].values
        pre_meds = ts.iloc[:-1, 6:10].values
        demo_info = ts.iloc[0, 2:6].values
        note_info = np.array(ts.iloc[:-1, -1].tolist())
        
        vitals = self.pad_sequences([vitals], self.max_len)[0]
        pre_meds = self.pad_sequences([pre_meds], self.max_len)[0]
        note_info = self.pad_sequences([note_info], self.max_len)[0]

        label = ts.iloc[-1, -3]

        vitals = torch.tensor(vitals, dtype=torch.float)
        pre_meds = torch.tensor(pre_meds, dtype=torch.float)
        note_info = torch.tensor(note_info, dtype=torch.float).squeeze()
        label = torch.tensor(label, dtype=torch.float)
        seq_len = torch.tensor(seq_len, dtype=torch.float)
        demo_info = torch.tensor(demo_info.astype(np.float64), dtype=torch.float)

        return vitals, pre_meds, note_info, demo_info, seq_len, label


class MortalityEHRPredDataset(Dataset):
    def __init__(self, data):
        with open(data, 'rb') as file:
            self.data = pickle.load(file)
        self.max_len = max(len(ts) for ts in self.data) - 1 

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pad_sequences(sequences, max_len, dtype='float32'):
        padded_seqs = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padded_seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]), dtype=dtype)])
            else:
                padded_seq = seq
            padded_seqs.append(padded_seq)
        
        return np.stack(padded_seqs)

    def __getitem__(self, idx):
        ts = self.data[idx]
        seq_len = len(ts) - 1

        vital_indices = list(range(10, ts.shape[1] - 3))
        vitals = ts.iloc[:-1, vital_indices].values
        pre_meds = ts.iloc[:-1, 6:10].values
        demo_info = ts.iloc[0, 2:6].values
        note_info = np.array(ts.iloc[:-1, -1].tolist())
        
        vitals = self.pad_sequences([vitals], self.max_len)[0]
        pre_meds = self.pad_sequences([pre_meds], self.max_len)[0]
        note_info = self.pad_sequences([note_info], self.max_len)[0]

        label = ts.iloc[-1, -3]
        mortality = ts.iloc[-1, -2]

        vitals = torch.tensor(vitals, dtype=torch.float)
        pre_meds = torch.tensor(pre_meds, dtype=torch.float)
        note_info = torch.tensor(note_info, dtype=torch.float).squeeze()
        label = torch.tensor(label, dtype=torch.float)
        seq_len = torch.tensor(seq_len, dtype=torch.float)
        demo_info = torch.tensor(demo_info.astype(np.float64), dtype=torch.float)

        return vitals, pre_meds, note_info, demo_info, seq_len, label, mortality

def mixedsample(dataset, fixed_size=1000):
    label_counts = Counter([dataset[i][-2].item() for i in range(len(dataset))])

    class_indices = {label: [] for label in label_counts}
    for i in range(len(dataset)):
        label = dataset[i][-2].item()
        class_indices[label].append(i)

    sampled_indices = []
    for label, indices in class_indices.items():
        if len(indices) > fixed_size:
            sampled_indices += random.sample(indices, fixed_size)
        elif len(indices) < fixed_size:
            sampled_indices += indices * (fixed_size // len(indices)) + indices[:fixed_size % len(indices)]
        else:
            sampled_indices += indices

    return Subset(dataset, sampled_indices)


# Test the dataloader
if __name__ == "__main__":
    #dataset = EHRPredDataset('sepsis_time_series_data.csv')
    #dataset = mixedsample(dataset, 1000)
    #dataloader = DataLoader(dataset, batch_size=40, shuffle=True)
    reliable_dataset = ReliableEHRPredDataset('sepsis_time_series_data.csv')
    reliable_dataloader = DataLoader(reliable_dataset, batch_size=40, shuffle=True)
    vitals, pre_meds, note_info, demo_info, seq_len, labels = next(iter(reliable_dataloader))
    print(vitals.shape)
    print(pre_meds.shape)
    print(note_info.shape)
    print(labels)
    print(seq_len)
    print(demo_info.shape)
    vitals, pre_meds, note_info, demo_info, seq_len, labels = vitals.to(torch.device('cuda:0')), pre_meds.to(torch.device('cuda:0')), note_info.to(torch.device('cuda:0')), demo_info.to(torch.device('cuda:0')), seq_len.to(torch.device('cuda:0')), labels.to(torch.device('cuda:0'))
    model = NextMedPredModel(vitals_feature_dim=45, num_classes=25, med_emb_dim=4, notes_emb_dim=768, d_model=64).to(torch.device('cuda:0'))
    refine_module = RefinePrediction(d_model=64, num_classes=25).to(torch.device('cuda:0'))
    embed, y_pred = model(vitals, pre_meds, note_info, demo_info, seq_len)
    y_refine = refine_module(embed, y_pred)
    print(y_refine[:2])
    loss = nn.MSELoss()(y_refine, y_pred)
    print(loss)






