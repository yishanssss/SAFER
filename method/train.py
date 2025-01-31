import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from utils import compute_accuracy, compute_auc, HR_at_k, MRR_at_k, compute_fdr
from model import NextMedPredModel, RefinePrediction
from dataloader import EHRPredDataset
import argparse
import numpy as np
from tqdm import tqdm, trange
from collections import Counter
from sklearn.metrics import roc_auc_score


def train_model(model, train_dataloader, val_dataloader, args, rank):
    device = torch.device(f'cuda:{rank}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in trange(args.epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for vitals, prev_meds, note_meds, demo, seq_len, labels in train_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device)
            optimizer.zero_grad()
            embed, outputs = model(vitals, prev_meds, note_meds, demo, seq_len)
            outputs = outputs.view(-1, outputs.size(-1))
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}')
        
        # Evaluate on the validation set after each epoch
        print('Evaluating on the validation set...')
        evaluate_model(model, val_dataloader, rank, args)


def evaluate_model(model, dataloader, rank, args):
    device = torch.device(f'cuda:{rank}')
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, labels in dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            all_preds.extend(y_pred.cpu())
            all_labels.extend(labels.detach().cpu().numpy())

    all_labels = np.array(all_labels, dtype=int)
    all_preds = torch.cat(all_preds, dim=0).view(len(all_preds), -1)  # Concatenate list of tensors into a single tensor

    #print(all_preds.shape, torch.tensor(all_labels).shape)
    accuracy = compute_accuracy(all_preds, all_labels)
    auc_macro, auc_micro = compute_auc(all_preds, all_labels, args.num_classes)
    hr_at_3 = HR_at_k(all_preds, all_labels, 3)
    mrr_at_3 = MRR_at_k(all_preds, all_labels, 3)
    fdr = compute_fdr(all_preds, all_labels, args.num_classes)
    print(f'Accuracy: {accuracy:.4f}, macro AUC: {auc_macro:.4f}, micro AUC: {auc_micro:.4f}, HR@3: {hr_at_3:.4f}, MRR@3: {mrr_at_3:.4f}, FDR: {fdr:.4f}')

def train_refine_module(model, refine_module, train_dataloader, val_dataloader, args, rank):
    device = torch.device(f'cuda:{rank}')
    mse = nn.MSELoss(reduction='mean')
    kl = nn.KLDivLoss(reduction='batchmean')
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(refine_module.parameters(), lr=args.lr_re)
    model.eval()
    refine_module.train()

    for epoch in trange(args.epochs_re):
        total_loss = 0
        for vitals, prev_meds, note_meds, demo, seq_len, labels in train_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            y_refine, _ = refine_module(embed, y_pred)
            loss = ce(y_refine, labels.long())
            #loss = args.alpha*mse(y_refine, y_pred)+(1-args.alpha)*kl(F.log_softmax(y_refine, dim=-1), F.softmax(y_pred, dim=-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}')
        
        # Evaluate on the validation set after each epoch
        print('Evaluating on the validation set...')
        evaluate_refine_module(model, refine_module, val_dataloader, rank, args)

def evaluate_refine_module(model, refine_module, dataloader, rank, args):
    device = torch.device(f'cuda:{rank}')
    mse = nn.MSELoss(reduction='mean')
    kl = nn.KLDivLoss(reduction='batchmean')
    ce = nn.CrossEntropyLoss()
    refine_module.eval()
    total_loss = 0
    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, labels in dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            y_refine, _ = refine_module(embed, y_pred)
            loss = ce(y_refine, labels.long())
            total_loss += loss.item()

    avg_val_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

def check_difference(model, refine_module, reliable_dataloader, uncertain_dataloader, rank):
    device = torch.device(f'cuda:{rank}')
    model.eval()
    refine_module.eval()
    average_diff_re = 0.0
    average_diff_un = 0.0
    min_re_diff = 1e6
    max_re_diff = -1e6
    min_un_diff = 1e6
    max_un_diff = -1e6
    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, labels in reliable_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            _, diff_re = refine_module(embed, y_pred)
            max_re_diff = max(diff_re.max(), max_re_diff)
            min_re_diff = min(diff_re.min(), min_re_diff)
            average_diff_re += torch.mean(diff_re)

    average_diff_re /= len(reliable_dataloader)

    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, labels in uncertain_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            _, diff_un = refine_module(embed, y_pred)
            max_un_diff = max(diff_un.max(), max_un_diff)
            min_un_diff = min(diff_un.min(), min_un_diff)
            average_diff_un += torch.mean(diff_un)

    average_diff_un /= len(uncertain_dataloader)

    max_diff = max(max_re_diff, max_un_diff)
    min_diff = min(min_re_diff, min_un_diff)
    print(f'Difference for reliable is {average_diff_re}, for uncertain is {average_diff_un}')

    return max_diff, min_diff


def uncerntain_aware_fine_tuning(model, refine_module, train_dataloader, val_dataloader, args, rank, max_diff, min_diff):
    device = torch.device(f'cuda:{rank}')
    mse = nn.MSELoss(reduction='mean')
    kl = nn.KLDivLoss(reduction='batchmean')
    ce = nn.CrossEntropyLoss(reduction='none')
    optimizer_model = optim.Adam(model.parameters(), lr=0.00001)
    optimizer_refine = optim.Adam(refine_module.parameters(), lr=0.00001)
    torch.autograd.set_detect_anomaly(True)


    for epoch in trange(args.epochs):
        model.train()
        refine_module.train()
        total_loss = 0
        for vitals, prev_meds, note_meds, demo, seq_len, labels in train_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device)
            
            optimizer_model.zero_grad()
            
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            y_refine, diff = refine_module(embed, y_pred)
            with torch.no_grad():
                diff = diff - min_diff / (max_diff - min_diff)
                diff = torch.clamp(diff, 0, 1)
                weights = 1 - diff + 1e-6                    #weights are inversely proportional to the difference between the prediction and the refined prediction
                weights = torch.clamp(weights, 0, 1)
                weights = torch.where(weights > 0.6, torch.tensor(1.0), weights)
            #print(weights)
            model_loss = ce(y_pred, labels.long()) * weights + args.gamma * (1-weights)     #Assign higher weights to reliable predictions
            model_loss = torch.mean(model_loss)
            #refine_loss = ce(y_refine, labels.long()) * weights
            #refine_loss = torch.mean(refine_loss)
            model_loss.backward()
            optimizer_model.step()

            total_loss += model_loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}')
        
        # Evaluate on the validation set after each epoch
        print('Evaluating on the validation set...')
        evaluate_model(model, val_dataloader, rank, args)

def train_mortality(model, mortality_model, train_dataloader, val_dataloader, args, rank):
    device = torch.device(f'cuda:{rank}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mortality_model.parameters(), lr=args.lr)

    for epoch in trange(args.epochs):
        model.eval()
        mortality_model.train()
        total_loss = 0
        for vitals, prev_meds, note_meds, demo, seq_len, labels, mortality in train_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels, mortality = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device), mortality.to(device)
            batch_size = vitals.size(0)
            optimizer.zero_grad()
            with torch.no_grad():
                embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            one_hot = F.one_hot(labels.long(), num_classes=args.num_classes)
            mortality_pred = mortality_model(vitals, note_meds, demo, one_hot)
            mortality_pred = mortality_pred.view(batch_size, -1)
            mortality = mortality.view(batch_size)
            loss = criterion(mortality_pred, mortality.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}')

        print('Evaluating on the validation set...')
        evaluate_mortality(model, mortality_model, val_dataloader, rank, args)

def evaluate_mortality(model, mortality_model, dataloader, rank, args):
    device = torch.device(f'cuda:{rank}')
    model.eval()
    mortality_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, labels, mortality in dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, labels, mortality = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), labels.to(device), mortality.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            mortality_pred = mortality_model(vitals, note_meds, demo, y_pred)
            all_preds.extend(mortality_pred.cpu())
            all_labels.extend(mortality.detach().cpu().numpy())

    all_labels = torch.tensor(all_labels, dtype=torch.float)
    all_preds = torch.cat(all_preds, dim=0).view(len(all_preds), -1)  # Concatenate list of tensors into a single tensor

    original_mortality = torch.sum(all_labels).item() / len(all_labels)
    pred_mortality = torch.sum(all_preds.argmax(dim=1)).item() / len(all_preds)
    print(f'Original Mortality: {original_mortality:.4f}, Predicted Mortality: {pred_mortality:.4f}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    dataset = EHRPredDataset('sepsis_time_series_data.csv')
    train_indices = np.random.choice(len(dataset), 200, replace=False)
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_indices = np.random.choice(len(dataset), 50, replace=False)
    eval_dataset = Subset(dataset, eval_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    model = NextMedPredModel(vitals_feature_dim=45, num_classes=4, med_emb_dim=4, notes_emb_dim=384, d_model=64).to(torch.device('cuda:0'))
    train_model(model, train_loader, eval_loader, args, 0)