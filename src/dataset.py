import sys
import os

# Προσθήκη του κεντρικού φακέλου CardioScan στο path αναζήτησης της Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from src.filters import preprocess_ecg


# Αντιστοίχιση ιατρικών διαγνώσεων του MIT-BIH σε αριθμητικές κλάσεις:
# 0: Normal Beat (Φυσιολογικός παλμός)
# 1: Premature Ventricular Contraction (Κοιλιακή έκτακτη συστολή - Αρρυθμία)
LABEL_MAPPING = {
    'N': 0,  # Normal
    'V': 1   # PVC (Arrhythmia)
}

class ECGDataset(Dataset):
    def __init__(self, records=['100', '101', '106'], window_before=100, window_after=150):
        """
        Κατεβάζει και επεξεργάζεται πραγματικά αρχεία ΗΚΓ από το PhysioNet.
        - window_before: 100 δείγματα (~0.28 sec πριν το R-peak)
        - window_after: 150 δείγματα (~0.42 sec μετά το R-peak)
        - Σύνολο διαστάσεων ανά παλμό: 250 σημεία
        """
        self.samples = []
        self.labels = []
        self.window_before = window_before
        self.window_after = window_after

        print("-> Φόρτωση πραγματικών δεδομένων από το PhysioNet MIT-BIH...")

        for rec in records:
            # Αυτόματη λήψη του σήματος και των ιατρικών σημειώσεων (annotations)
            record = wfdb.rdrecord(rec, pn_dir='mitdb')
            annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')

            # Παίρνουμε το κύριο κανάλι (MLII) και το καθαρίζουμε με τα φίλτρα μας
            raw_signal = record.p_signal[:, 0]
            clean_signal = preprocess_ecg(raw_signal, fs=record.fs)

            r_peaks = annotation.sample
            symbols = annotation.symbol

            # Τεμαχισμός του σήματος γύρω από κάθε διαγνωσμένο R-peak
            for r_pos, sym in zip(r_peaks, symbols):
                if sym in LABEL_MAPPING:
                    start = r_pos - self.window_before
                    end = r_pos + self.window_after

                    # Έλεγχος ορίων ώστε να μην βγούμε εκτός μήκους σήματος
                    if start >= 0 and end < len(clean_signal):
                        beat = clean_signal[start:end]
                        self.samples.append(beat)
                        self.labels.append(LABEL_MAPPING[sym])

        # Μετατροπή σε PyTorch Tensors
        # Διάσταση εισόδου για 1D-CNN: (Αριθμός παλμών, 1 Κανάλι, 250 Σημεία)
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

        print(f"-> Ολοκληρώθηκε! Φορτώθηκαν {len(self.samples)} παλμοί.")
        print(f"   - Φυσιολογικοί (Normal - 0): {torch.sum(self.labels == 0).item()}")
        print(f"   - Αρρυθμίες (PVC - 1): {torch.sum(self.labels == 1).item()}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# Δοκιμαστική εκτέλεση του dataset
if __name__ == '__main__':
    dataset = ECGDataset(records=['100', '106'])
    sample_x, sample_y = dataset[0]
    print(f"-> Διαστάσεις ενός δείγματος εισόδου (Tensor Shape): {sample_x.shape}")
