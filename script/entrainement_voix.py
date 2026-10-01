import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers

# =========================================================
# PARAMÈTRES AUDIO ET ENTRAÎNEMENT
# =========================================================

DUREE_AUDIO = 2.0  # Durée fixe en secondes pour chaque extrait
SAMPLE_RATE = 22050
N_MFCC = 40        # Nombre de coefficients MFCC

BATCH_SIZE = 16
EPOCHS = 60

CLASSES = ['moi', 'ami', 'autre']

# =========================================================
# FONCTION DE PRÉTRAITEMENT AUDIO (AUDIO -> MFCC)
# =========================================================

def charger_et_extraire_mfcc(chemin_fichier, target_duration=DUREE_AUDIO, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    y, sr = librosa.load(chemin_fichier, sr=sr)
    
    target_len = int(sr * target_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
        
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Normalisation globale
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc

def charger_dataset_depuis_dossier(dossier_base):
    X, y = [], []
    extensions_valides = ('.mp3', '.wav', '.ogg', '.flac', '.m4a')
    
    if not os.path.exists(dossier_base):
        print(f"⚠️ ATTENTION : Le dossier {dossier_base} n'existe pas !")
        return np.array(X), np.array(y)

    for index_classe, nom_classe in enumerate(CLASSES):
        dossier_classe = os.path.join(dossier_base, nom_classe)
        if not os.path.exists(dossier_classe):
            print(f"⚠️ ATTENTION : Sous-dossier introuvable : {dossier_classe}")
            continue
            
        fichiers = os.listdir(dossier_classe)
        compteur = 0
        
        for fichier in fichiers:
            if fichier.lower().endswith(extensions_valides):
                chemin = os.path.join(dossier_classe, fichier)
                try:
                    mfcc = charger_et_extraire_mfcc(chemin)
                    X.append(mfcc)
                    y.append(index_classe)
                    compteur += 1
                except Exception as e:
                    print(f"Erreur de lecture sur {fichier} : {e}")
                    
        print(f"  -> Classe '{nom_classe}': {compteur} fichiers chargés")
                    
    return np.array(X), np.array(y)

# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================

CHEMIN_ENTRAINEMENT = "/home/ubuntu/Reconnaissance_Vocale/Donnees/Entrainement"
CHEMIN_TEST = "/home/ubuntu/Reconnaissance_Vocale/Donnees/Test"

print("Chargement des données d'entraînement...")
X_train, y_train = charger_dataset_depuis_dossier(CHEMIN_ENTRAINEMENT)

print("\nChargement des données de test...")
X_test, y_test = charger_dataset_depuis_dossier(CHEMIN_TEST)

if len(X_train) == 0:
    raise ValueError("\n❌ Aucun fichier audio n'a pu être chargé. Vérifiez vos chemins d'accès.")

print(f"\nForme des entrées (X_train) : {X_train.shape}")
print(f"Classes détectées : {CLASSES}")

# =========================================================
# MODÈLE RNA / MLP (PERCEPTRON MULTI-COUCHES)
# =========================================================

input_shape = (X_train.shape[1], X_train.shape[2]) # ex: (40, 87)

modele_rna = tf.keras.Sequential([
    layers.Input(shape=input_shape),
    
    # Applatissement des MFCC (40x87 -> 3480 entrées)
    layers.Flatten(),
    
    # Couche cachée 1
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Couche cachée 2
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Couche cachée 3
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # Couche de sortie RNA (Softmax)
    layers.Dense(3, activation='softmax')
])

# =========================================================
# COMPILATION
# =========================================================

modele_rna.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n===== RÉSUMÉ DU MODÈLE RNA (MLP) =====")
modele_rna.summary()

# =========================================================
# ENTRAÎNEMENT
# =========================================================

print("\n===== ENTRAÎNEMENT RNA =====")
historique = modele_rna.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# =========================================================
# ÉVALUATION ET SAUVEGARDE
# =========================================================

print("\n===== ÉVALUATION =====")
loss, accuracy = modele_rna.evaluate(X_test, y_test)
print(f"\nLoss      : {loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")

CHEMIN_MODELE = "/home/ubuntu/Reconnaissance_Vocale/modele/modele_reconnaissance_vocale.h5"
os.makedirs(os.path.dirname(CHEMIN_MODELE), exist_ok=True)
modele_rna.save(CHEMIN_MODELE)

print(f"\nModèle RNA sauvegardé : {CHEMIN_MODELE}")