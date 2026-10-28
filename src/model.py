import torch
import torch.nn as nn

class CardioNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(CardioNet1D, self).__init__()
        
        # 1ο Συνελικτικό Block: Εξαγωγή βασικών κυματομορφών (μεγάλο kernel=7 για ευρύ χρονικό παράθυρο)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Μειώνει το μήκος από 250 -> 125
        )
        
        # 2ο Συνελικτικό Block: Ανίχνευση σύνθετων μορφολογιών του QRS
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Μειώνει το μήκος από 125 -> 62
        )
        
        # 3ο Συνελικτικό Block: Υψηλού επιπέδου χαρακτηριστικά αρρυθμίας
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Μειώνει το μήκος από 62 -> 31
        )
        
        # Global Average Pooling: Συμπυκνώνει κάθε κανάλι σε μία τιμή (ιδανικό για Explainability / Grad-CAM)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Πλήρως Διασυνδεδεμένα Επίπεδα (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),  # Αποφυγή overfitting
            nn.Linear(32, num_classes)  # Έξοδος: 2 κλάσεις (Normal vs Arrhythmia)
        )

    def forward(self, x):
        # Είσοδος: (Batch_Size, Channels=1, Length=250)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Global pooling & Flattening
        x = self.global_pool(x)       # Shape: (Batch_Size, 64, 1)
        x = torch.flatten(x, 1)        # Shape: (Batch_Size, 64)
        
        logits = self.classifier(x)   # Shape: (Batch_Size, 2)
        return logits


if __name__ == '__main__':
    # Δοκιμή της αρχιτεκτονικής με ένα τυχαίο δείγμα (batch=4 παλμοί, 1 κανάλι, 250 σημεία)
    dummy_input = torch.randn(4, 1, 250)
    model = CardioNet1D(in_channels=1, num_classes=2)
    output = model(dummy_input)
    
    # Υπολογισμός συνολικών παραμέτρων
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-> Το μοντέλο CardioNet1D αρχικοποιήθηκε επιτυχώς!")
    print(f"-> Είσοδος Tensor Shape:  {dummy_input.shape}")
    print(f"-> Έξοδος Logits Shape:   {output.shape} (4 παλμοί x 2 κλάσεις)")
    print(f"-> Συνολικές Εκπαιδεύσιμες Παράμετροι: {total_params:,}")
