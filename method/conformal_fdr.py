import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Subset
import itertools


def sigmoid(uncertainty_score):
    """
    Converts an uncertainty score to a probability using the sigmoid function.
    """
    return 1 / (1 + np.exp(-uncertainty_score))

def nonconformalized_score(model, refine_module, train_dataloader, val_dataloader, test_dataloader, max_diff, min_diff, args, rank):
    device = torch.device(f'cuda:{rank}')
    model.eval()
    refine_module.eval()
    emb_calib = []
    calibration_scores = []

    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, label in train_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, label = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), label.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            y_refine, diff = refine_module(embed, y_pred)
            diff = diff - min_diff / (max_diff - min_diff)
            #diff = torch.where(diff < 0.5, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
            diff = torch.clamp(diff, 0, 1)
            calibration_scores.append(diff.cpu().numpy())
            emb_calib.append(embed.cpu().numpy())

    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, label in val_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, label = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), label.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            y_refine, diff = refine_module(embed, y_pred) ## diff -> KL divergence >0
            diff = diff - min_diff / (max_diff - min_diff)
            #diff = torch.where(diff < 0.5, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
            diff = torch.clamp(diff, 0, 1)
            calibration_scores.append(diff.cpu().numpy())
            emb_calib.append(embed.cpu().numpy())

    calibration_scores = np.concatenate(calibration_scores)
    emb_calib = np.concatenate(emb_calib)
    #np.save('calibration_scores.npy', calibration_scores)
    #np.save('calibration_embeddings.npy', emb_calib)

    p_values = []
    labels = []
    predictions = []
    emb_test = []
    test_diff = []
    with torch.no_grad():
        for vitals, prev_meds, note_meds, demo, seq_len, label in test_dataloader:
            vitals, prev_meds, note_meds, demo, seq_len, label = vitals.to(device), prev_meds.to(device), note_meds.to(device), demo.to(device), seq_len.to(device), label.to(device)
            embed, y_pred = model(vitals, prev_meds, note_meds, demo, seq_len)
            y_refine, diff = refine_module(embed, y_pred)
            diff = diff - min_diff / (max_diff - min_diff)
            #diff = torch.where(diff < 0.5, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
            diff = torch.clamp(diff, 0, 1)
            test_diff.append(diff.cpu().numpy())
            labels.append(label.cpu().numpy())
            predictions.append(torch.argmax(y_pred, dim=1).cpu().numpy())
            emb_test.append(embed.cpu().numpy())

            for d in diff.cpu().numpy():
                #print(np.sum(calibration_scores <= d))
                #print("selected numbers in p-value:")
                #print("d:" )
                #print(d)
                #print(np.sum((calibration_scores >= args.c) & (calibration_scores <= d)))
                U_j = np.random.uniform(0, 1)  # Uniform random variable U_j in [0, 1]

                # Calculate p-value based on the formula
                indicator_1 = (calibration_scores < d) & (calibration_scores >= args.c)
                indicator_2 = (calibration_scores == d) & (calibration_scores >= args.c)

                numerator = (
                    np.sum(indicator_1) +
                    U_j * (1 + np.sum(indicator_2)))
                p_value = numerator / (1 + len(calibration_scores))
                p_values.append(p_value)

    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    test_diff = np.concatenate(test_diff)
    emb_test = np.concatenate(emb_test)
    #np.save('test_scores.npy', test_diff)
    #np.save('test_embeddings.npy', emb_test)
    #print(test_diff)

    sorted_p_values = sorted(p_values)
    max_r = 0

    for r, p_value in enumerate(sorted_p_values):
        if p_value > (r + 1) * args.alpha / len(sorted_p_values):
            break
        else:
            max_r = r + 1
    print('max_r:')
    print(max_r)

    if max_r == 0:
        filtered_indices = []
    else:
        p_threshold = sorted_p_values[max_r-1]
        filtered_indices = [i for i, p_value in enumerate(p_values) if p_value <= p_threshold]
    print('filtered_indices:')
    print(filtered_indices)
    filtered_labels = [labels[i] for i in filtered_indices]
    filtered_predictions = [predictions[i] for i in filtered_indices]
    filtered_diff = np.array([test_diff[i] for i in filtered_indices])

    fdr = np.sum(filtered_diff >= args.c) / max(1, len(filtered_diff))
    power = np.sum(filtered_diff < args.c) / max(1, np.sum(test_diff < args.c))

    '''
    fdr_list = []
    total_tp = 0
    total_fp = 0
    for class_id in range(args.num_classes):
        true_positives = np.sum(np.logical_and(np.array(filtered_labels) == class_id, np.array(filtered_predictions) == class_id))
        false_positives = np.sum(np.logical_and(np.array(filtered_labels) != class_id, np.array(filtered_predictions) == class_id))
        fdr = false_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        fdr_list.append(fdr)
        total_tp += true_positives
        total_fp += false_positives
    
    fdr = np.mean(fdr_list)
    '''
    print(f'FDR at alpha={args.alpha}: {fdr:.4f}')
    print(f'Power at alpha={args.alpha}: {power:.4f}')

    return emb_calib, calibration_scores, emb_test, test_diff


class SampledDataLoader:
    def __init__(self, dataloader, fraction=0.1):
        self.dataloader = dataloader
        self.fraction = fraction
        self.sampled_size = int(len(dataloader) * fraction)
    
    def __iter__(self):
        batch_iterator = iter(self.dataloader)
        sampled_batches = itertools.islice(batch_iterator, self.sampled_size)
        return sampled_batches
    
    def __len__(self):
        return self.sampled_size