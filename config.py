"""
config.py — Configuration centralisée de l'application DermAI
==============================================================
Toutes les constantes et paramètres sont définis ici.
Les valeurs sensibles (mots de passe, clés) doivent être
définies dans un fichier .env (non versionné sur GitHub).

Usage : app.config.from_object(Config)
"""

import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()


class Config:
    # ── Sécurité Flask ──────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY", "dermai_secret_key_enstab_2026")

    # ── Base de données MySQL ───────────────────────────────────────────────
    DB_HOST     = os.environ.get("DB_HOST",     "localhost")
    DB_USER     = os.environ.get("DB_USER",     "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_NAME     = os.environ.get("DB_NAME",     "dermai_db")

    # ── Modèle IA ───────────────────────────────────────────────────────────
    MODEL_PATH = os.path.join("model", "vgg16_skin_cancer.h5")

    # Paramètres VGG16 (Transfer Learning — TD7)
    IMG_WIDTH       = 224
    IMG_HEIGHT      = 224
    PREDICTION_THRESHOLD = 0.5  # Seuil sigmoid : >0.5 → Malin

    # ── Upload ──────────────────────────────────────────────────────────────
    UPLOAD_FOLDER      = os.path.join("static", "uploads")
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 Mo maximum

    # ── Mode Debug ──────────────────────────────────────────────────────────
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
