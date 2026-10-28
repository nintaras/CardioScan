import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.filters import preprocess_ecg

# 1. Παράμετροι σήματος
fs = 360.0  # Συχνότητα δειγματοληψίας (360 δείγματα / δευτερόλεπτο)
duration = 5.0  # Διάρκεια 5 δευτερολέπτων
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 2. Προσομοίωση καθαρού καρδιακού παλμού (περίπου 72 παλμοί ανά λεπτό -> 1.2 Hz)
clean_ecg = np.sin(2 * np.pi * 1.2 * t) ** 5

# 3. Προσθήκη πραγματικών πηγών βιοϊατρικού θορύβου:
# - Baseline wander: Χαμηλόσυχνη ταλάντωση (0.2 Hz) λόγω αναπνοής
baseline_wander = 0.8 * np.sin(2 * np.pi * 0.2 * t)
# - Powerline interference: Έντονος θόρυβος δικτύου 50 Hz
powerline_50hz = 0.35 * np.sin(2 * np.pi * 50.0 * t)
# - Μυϊκός τυχαίος θόρυβος (λευκός θόρυβος)
random_noise = np.random.normal(0, 0.05, len(t))

# Το τελικό ακατέργαστο σήμα που "λαμβάνει" ο αισθητήρας
raw_ecg = clean_ecg + baseline_wander + powerline_50hz + random_noise

# 4. Εφαρμογή των φίλτρων που φτιάξαμε στο src/filters.py
filtered_ecg = preprocess_ecg(raw_ecg, fs=fs)

# 5. Εντοπισμός των κορυφών R (R-Peaks) στο καθαρό σήμα
# Ελάχιστη απόσταση μεταξύ παλμών: 200 ms (fs * 0.2)
peaks, _ = find_peaks(filtered_ecg, distance=int(fs * 0.4), height=1.0)

# 6. Σχεδιασμός του γραφήματος
plt.figure(figsize=(12, 7))

# Πάνω γράφημα: Ακατέργαστο σήμα
plt.subplot(2, 1, 1)
plt.plot(t, raw_ecg, color='#dc2626', linewidth=1, label='Raw ECG (με θόρυβο αναπνοής & 50Hz)')
plt.title('1. Ακατέργαστο Σήμα ΗΚΓ (Noisy Input)', fontsize=12, fontweight='bold')
plt.ylabel('Τάση (mV)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

# Κάτω γράφημα: Καθαρισμένο σήμα + Εντοπισμός R-Peaks
plt.subplot(2, 1, 2)
plt.plot(t, filtered_ecg, color='#0284c7', linewidth=1.2, label='Filtered ECG (Butterworth + Notch + Z-Score)')
plt.plot(t[peaks], filtered_ecg[peaks], 'ro', markersize=6, label=f'R-Peaks ({len(peaks)} παλμοί)')
plt.title('2. Καθαρισμένο Σήμα & Εντοπισμός R-Peaks (Clean Output)', fontsize=12, fontweight='bold')
plt.xlabel('Χρόνος (δευτερόλεπτα)')
plt.ylabel('Κανονικοποιημένο Πλάτος')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

plt.tight_layout()

# Αποθήκευση εικόνας για το portfolio / README
plt.savefig('filtering_result.png', dpi=150)
print(f"-> Επιτυχία! Εντοπίστηκαν {len(peaks)} καρδιακοί παλμοί.")
print("-> Το γράφημα αποθηκεύτηκε ως 'filtering_result.png'.")

# Εμφάνιση του παραθύρου στην οθόνη
plt.show()
