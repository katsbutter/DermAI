"""
scripts/train_model.py — Entraînement du modèle VGG16 (Transfer Learning)
==========================================================================
Script d'entraînement documenté, conforme au TD7 du cours Introduction à l'IA.

Architecture :
  - Modèle de base   : VGG16 pré-entraîné sur ImageNet (couches gelées)
  - Têtes ajoutées   : Flatten → Dropout(0.5) → Dense(256, ReLU) → Dense(1, Sigmoid)
  - Optimiseur       : Adam (lr=0.0001)
  - Fonction de perte: binary_crossentropy
  - Dataset          : Kaggle — Skin Cancer Malignant vs Benign

Dataset Kaggle :
  kaggle datasets download -d fanconic/skin-cancer-malignant-vs-benign
  unzip skin-cancer-malignant-vs-benign.zip -d data/

Structure attendue :
  data/
  ├── train/
  │   ├── benign/
  │   └── malignant/
  └── test/
      ├── benign/
      └── malignant/

Résultat : model/vgg16_skin_cancer.h5

Cours : Introduction à l'IA — ENSTAB | Dr. Amira Echtioui
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Paramètres ──────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4

TRAIN_DIR  = os.path.join("data", "train")
TEST_DIR   = os.path.join("data", "test")
MODEL_PATH = os.path.join("model", "vgg16_skin_cancer.h5")

os.makedirs("model", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRÉPARATION DES DONNÉES (augmentation + normalisation)
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ÉTAPE 1 : Préparation des données")
print("=" * 60)

# Générateur pour l'entraînement (avec augmentation)
train_datagen = ImageDataGenerator(
    rescale           = 1.0 / 255,   # Normalisation [0,1]
    rotation_range    = 25,
    width_shift_range = 0.2,
    height_shift_range= 0.2,
    shear_range       = 0.2,
    zoom_range        = 0.2,
    horizontal_flip   = True,
    fill_mode         = "nearest",
)

# Générateur pour le test (normalisation uniquement)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = "binary",
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = "binary",
    shuffle     = False,
)

print(f"\nClasses détectées : {train_generator.class_indices}")
print(f"  → 0 = Benign | 1 = Malignant")


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSTRUCTION DU MODÈLE (Transfer Learning VGG16)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ÉTAPE 2 : Construction du modèle VGG16 (Transfer Learning)")
print("=" * 60)

# Charger VGG16 sans la tête FC, poids ImageNet
base_model = VGG16(
    weights      = "imagenet",
    include_top  = False,
    input_shape  = (224, 224, 3),
)

# Geler toutes les couches du modèle de base
for layer in base_model.layers:
    layer.trainable = False

print(f"VGG16 chargé — {len(base_model.layers)} couches gelées")

# Ajouter les couches personnalisées (tête de classification)
x = Flatten()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer = Adam(learning_rate=LR),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"],
)

model.summary()


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ÉTAPE 3 : Entraînement")
print("=" * 60)

history = model.fit(
    train_generator,
    epochs           = EPOCHS,
    validation_data  = test_generator,
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SAUVEGARDE DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

model.save(MODEL_PATH)
print(f"\n[✓] Modèle sauvegardé : {MODEL_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. ÉVALUATION ET MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ÉTAPE 4 : Évaluation sur le jeu de test")
print("=" * 60)

test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"Accuracy sur test : {test_acc:.4f}")
print(f"Loss sur test     : {test_loss:.4f}")

# Prédictions pour le rapport de classification
y_true  = test_generator.classes
y_proba = model.predict(test_generator, verbose=0).ravel()
y_pred  = (y_proba > 0.5).astype(int)

print("\nRapport de classification :")
print(classification_report(
    y_true, y_pred,
    target_names=["Benign", "Malignant"],
))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de Confusion — VGG16 Transfer Learning")
plt.tight_layout()
plt.savefig(os.path.join("static", "images", "confusion_matrix.png"), dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 6. COURBES D'APPRENTISSAGE
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history["accuracy"],     label="Entraînement")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Évolution de l'Accuracy")
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history["loss"],     label="Entraînement")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Évolution de la Loss")
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    "Courbes d'Apprentissage — VGG16 Transfer Learning",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join("static", "images", "learning_curves.png"), dpi=150)
plt.show()

print("\n[✓] Entraînement terminé. Modèle prêt pour l'application web.")
