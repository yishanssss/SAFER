import torch
import torch.nn as nn
import torch.nn.functional as F

class MortalityRate(nn.Module):
    def __init__(self, input_dim, hidden_dim, vitals_feature_dim, notes_emb_dim, demo_dim):
        super(MortalityRate, self).__init__()

        self.vitals_lin = nn.Sequential(
            nn.Linear(vitals_feature_dim, hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim),
        )
        self.lstm_vitals = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        
        self.notes_lin = nn.Sequential(
            nn.Linear(notes_emb_dim, hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
        self.lstm_notes = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)

        self.demo_lin = nn.Sequential(
            nn.Linear(demo_dim, hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim),
        )

        self.fc1 = nn.Linear(input_dim, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc3 = nn.Sequential(
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.Linear(2*hidden_dim, 2),
        )

    def forward(self, vitals, notes, demo, y_pred): #x: patitent embedding, y: medication
        vitals = F.relu(self.vitals_lin(vitals))
        notes = F.relu(self.notes_lin(notes))
        vitals, _ = self.lstm_vitals(vitals)
        vitals = vitals[:, -1, :].squeeze()
        notes, _ = self.lstm_notes(notes)
        notes = notes[:, -1, :].squeeze()

        demo = F.relu(self.demo_lin(demo))
        
        y_pred = y_pred.float()
        y_pred = F.relu(self.fc1(y_pred))
        y_pred = F.relu(self.fc2(y_pred))

        vitals, notes = vitals.view_as(y_pred), notes.view_as(y_pred)
        x = torch.cat([vitals, notes, demo, y_pred], dim=1)
        x = self.fc3(x)
        return x
