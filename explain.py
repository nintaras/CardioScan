import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import ECGDataset
from src.model import CardioNet1D

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks για την εξαγωγή των activations και των gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Global Average Pooling στα gradients
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        # Γραμμικός συνδυασμός βαρών και activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Εφαρμογή ReLU (μας ενδιαφέρουν μόνο οι θετικές συνεισφορές)
        cam = F.relu(cam)
        
        # Interpolate στο αρχικό μήκος του σήματος (250 σημεία)
        cam = F.interpolate(cam, size=input_tensor.shape[2], mode='linear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Κανονικοποίηση στο διάστημα [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, output.softmax(dim=1).detach().cpu().numpy()[0]


def run_explainability():
    # 1. Φόρτωση εκπαιδευμένου μοντέλου
    device = torch.device("cpu")  # Για inference και visualization η CPU είναι άμεση
    model = CardioNet1D(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load('best_cardioscan_model.pth', map_location=device))
    model.to(device)

    # Στοχεύουμε το τελευταίο συνελικτικό επίπεδο πριν το pooling
    target_layer = model.conv_block3[0]
    grad_cam = GradCAM1D(model, target_layer)

    # 2. Φόρτωση δεδομένων και εύρεση ενός παλμού αρρυθμίας (PVC - Class 1)
    dataset = ECGDataset(records=['100', '106'])
    
    pvc_indices = (dataset.labels == 1).nonzero(as_tuple=True)[0]
    sample_idx = pvc_indices[0].item()  # Επιλογή του 1ου εντοπισμένου παλμού αρρυθμίας
    
    ecg_signal, true_label = dataset[sample_idx]
    input_tensor = ecg_signal.unsqueeze(0).to(device)  # Shape: (1, 1, 250)

    # 3. Παραγωγή του Grad-CAM Heatmap
    heatmap, probabilities = grad_cam.generate_heatmap(input_tensor, target_class=1)
    
    raw_signal = ecg_signal.squeeze().numpy()
    time_ms = np.linspace(-100 * (1000/360), 150 * (1000/360), len(raw_signal))

    # 4. Σχεδιασμός Γραφήματος Εξηγησιμότητας
    plt.figure(figsize=(12, 6))
    
    # Χρωματισμός του σήματος ανάλογα με τη σημασία που έδωσε το AI
    plt.plot(time_ms, raw_signal, color='black', alpha=0.3, linewidth=1.5, label='ΗΚΓ Παλμός (PVC)')
    scatter = plt.scatter(time_ms, raw_signal, c=heatmap, cmap='jet', s=35, zorder=3, label='AI Attention (Grad-CAM)')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Βαθμός Συσχέτισης με Αρρυθμία (0.0 - 1.0)', fontsize=10)

    plt.axvline(0, color='gray', linestyle='--', alpha=0.7, label='R-Peak Center')
    plt.title(f"CardioScan XAI: Ερμηνεία Πρόβλεψης Αρρυθμίας (PVC)\n"
              f"Πιθανότητα Μοντέλου για Αρρυθμία: {probabilities[1]*100:.2f}% | Πραγματική Κλάση: Arrhythmia",
              fontsize=12, fontweight='bold')
    plt.xlabel('Χρόνος σε σχέση με το R-Peak (ms)')
    plt.ylabel('Κανονικοποιημένη Τάση')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('gradcam_arrhythmia.png', dpi=150)
    print("-> Το γράφημα εξηγησιμότητας δημιουργήθηκε και αποθηκεύτηκε ως 'gradcam_arrhythmia.png'!")
    plt.show()

if __name__ == '__main__':
    run_explainability()
