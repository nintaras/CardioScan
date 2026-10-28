import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=3):
    """
    Εφαρμόζει ζωνοπερατό φίλτρο Butterworth μηδενικής φάσης (filtfilt).
    - lowcut (0.5 Hz): Αποκόπτει την ολίσθηση γραμμής βάσης (αναπνοή ασθενούς).
    - highcut (45.0 Hz): Αποκόπτει υψίσυχνο μυϊκό θόρυβο (ΗΜΓ / τρόμος).
    - fs: Συχνότητα δειγματοληψίας (360 Hz, το πρότυπο του MIT-BIH dataset).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Χρήση filtfilt για να μην υπάρξει χρονική μετατόπιση φάσης στο σήμα
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def notch_filter(data, cutoff=50.0, fs=360.0, q=30.0):
    """
    Αφαιρεί στενά τον θόρυβο του δικτύου ηλεκτροδότησης (50 Hz).
    - q (Quality factor): Καθορίζει πόσο στενό είναι το εύρος αποκοπής γύρω από τα 50 Hz.
    """
    nyquist = 0.5 * fs
    freq = cutoff / nyquist
    b, a = iirnotch(freq, q)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def preprocess_ecg(raw_signal, fs=360.0):
    """
    Ολοκληρωμένη αλυσίδα επεξεργασίας:
    1. Bandpass φιλτράρισμα (0.5 - 45 Hz)
    2. Notch φιλτράρισμα στα 50 Hz
    3. Κανονικοποίηση Z-Score (μέση τιμή 0, τυπική απόκλιση 1)
    """
    # 1. Καθαρισμός χαμηλών & υψηλών συχνοτήτων
    signal_bp = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=45.0, fs=fs)
    
    # 2. Αφαίρεση θορύβου ρεύματος
    signal_clean = notch_filter(signal_bp, cutoff=50.0, fs=fs)
    
    # 3. Z-score normalization για ομαλή είσοδο στο νευρωνικό δίκτυο
    mean = np.mean(signal_clean)
    std = np.std(signal_clean)
    normalized_signal = (signal_clean - mean) / (std + 1e-8)
    
    return normalized_signal
