import numpy as np
import tensorflow as tf
import librosa

# =========================================================
# PARAMÈTRES
# =========================================================

CLASSES = ['Moi', 'Ami', 'Autre']
DUREE_AUDIO = 2.0
SAMPLE_RATE = 22050
N_MFCC = 40

# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================

CHEMIN_MODELE = "/home/ubuntu/Reconnaissance_Vocale/modele/modele_reconnaissance_vocale.h5"
modele = tf.keras.models.load_model(CHEMIN_MODELE)

# =========================================================
# PRÉTRAITEMENT DE L'AUDIO À TESTER
# =========================================================

CHEMIN_AUDIO = "/home/ubuntu/Reconnaissance_Vocale/Donnees/Test/moi/moi15.mp3"

def charger_et_extraire_mfcc(chemin_fichier):
    y, sr = librosa.load(chemin_fichier, sr=SAMPLE_RATE)
    target_len = int(sr * DUREE_AUDIO)
    
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
        
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc

# Extrait les MFCC et ajoute la dimension batch
mfcc_audio = charger_et_extraire_mfcc(CHEMIN_AUDIO)
mfcc_batch = np.expand_dims(mfcc_audio, axis=0)

# =========================================================
# PRÉDICTION
# =========================================================

prediction = modele.predict(mfcc_batch)
index_pred = np.argmax(prediction[0])
proba = prediction[0][index_pred] * 100

print("\n===== PRÉDICTION =====")
print(f"Probabilités brutes : {prediction[0]}")
print(f"Résultat : {CLASSES[index_pred]} (Confiance : {proba:.2f}%)")