"""
DermAI — Application Web de Diagnostic de Cancer Cutané
========================================================
Application Flask intégrant un modèle VGG16 pré-entraîné (Transfer Learning)
pour la classification de lésions cutanées en Bénin / Malin.

Cours : Introduction à l'IA — ENSTAB | Dr. Amira Echtioui
Auteur : [Votre Nom]
Année  : 2025/2026
"""

import os
import uuid
from functools import wraps
from datetime import datetime

import numpy as np
from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash
)
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash

from config import Config
from database import db_connect, init_db

# ── TensorFlow / Keras ──────────────────────────────────────────────────────
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# ═══════════════════════════════════════════════════════════════════════════
# INITIALISATION DE L'APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config.from_object(Config)

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Chargement du modèle IA ─────────────────────────────────────────────────
MODEL = None
try:
    MODEL = load_model(app.config["MODEL_PATH"])
    print(f"[✓] Modèle chargé depuis : {app.config['MODEL_PATH']}")
except Exception as e:
    print(f"[✗] Impossible de charger le modèle : {e}")
    print("    → L'application fonctionnera sans le modèle IA.")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    """Vérifie que l'extension du fichier est autorisée."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def login_required(f):
    """Décorateur : redirige vers /login si l'utilisateur n'est pas connecté."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Veuillez vous connecter pour accéder à cette page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pipeline de prétraitement pour VGG16 (identique à l'entraînement) :
    1. Chargement et redimensionnement en 224×224
    2. Conversion en tableau NumPy
    3. Normalisation des pixels dans [0, 1]
    4. Ajout de la dimension batch → (1, 224, 224, 3)
    """
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def run_prediction(image_path: str) -> dict:
    """
    Exécute la prédiction sur une image.

    Retourne :
        {
          "label"      : "Malin" | "Bénin",
          "label_en"   : "Malignant" | "Benign",
          "confidence" : float (0.0 – 100.0),
          "risk_level" : "Élevé" | "Faible",
          "color_class": "danger" | "success"
        }
    """
    if MODEL is None:
        raise RuntimeError(
            "Modèle non chargé. Vérifiez le chemin dans config.py."
        )

    img_array = preprocess_image(image_path)
    prediction = float(MODEL.predict(img_array, verbose=0)[0][0])

    if prediction > 0.5:
        return {
            "label"      : "Malin",
            "label_en"   : "Malignant",
            "confidence" : round(prediction * 100, 2),
            "risk_level" : "Élevé",
            "color_class": "danger",
        }
    else:
        return {
            "label"      : "Bénin",
            "label_en"   : "Benign",
            "confidence" : round((1 - prediction) * 100, 2),
            "risk_level" : "Faible",
            "color_class": "success",
        }


def get_stats() -> dict:
    """Récupère les statistiques globales depuis la base de données."""
    conn = db_connect()
    if conn is None:
        return {"total": 0, "malignant": 0, "benign": 0}
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) AS total FROM patients")
        total = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM patients WHERE result = 'Malignant'"
        )
        malignant = cursor.fetchone()["cnt"]

        return {
            "total"    : total,
            "malignant": malignant,
            "benign"   : total - malignant,
        }
    finally:
        cursor.close()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# ── Racine → redirection ────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


# ── Authentification ────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = db_connect()
        if conn is None:
            flash("Erreur de connexion à la base de données.", "danger")
            return render_template("login.html")

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM users WHERE username = %s", (username,)
            )
            user = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()

        # Vérification du mot de passe (hashé avec werkzeug)
        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            flash(f"Bienvenue, {user['username']} !", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Identifiants incorrects. Veuillez réessayer.", "danger")

    return render_template("login.html")


# ── Tableau de bord ─────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    stats = get_stats()
    return render_template(
        "dashboard.html",
        username=session.get("username"),
        stats=stats,
        model_loaded=(MODEL is not None),
    )


# ── Analyse / Prédiction ────────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        # Validation du fichier
        if "image" not in request.files:
            flash("Aucun fichier reçu.", "warning")
            return redirect(url_for("predict"))

        file = request.files["image"]

        if file.filename == "":
            flash("Veuillez sélectionner une image.", "warning")
            return redirect(url_for("predict"))

        if not allowed_file(file.filename):
            flash(
                "Format non supporté. Utilisez : JPG, JPEG ou PNG.", "warning"
            )
            return redirect(url_for("predict"))

        # Informations patient
        patient_name = request.form.get("name", "Inconnu").strip() or "Inconnu"
        patient_age  = request.form.get("age", "").strip()

        try:
            patient_age = int(patient_age) if patient_age.isdigit() else None
        except ValueError:
            patient_age = None

        try:
            # Sauvegarde sécurisée du fichier avec nom unique
            ext      = secure_filename(file.filename).rsplit(".", 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Prédiction IA
            result = run_prediction(filepath)

            # Sauvegarde en base de données
            conn = db_connect()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO patients
                            (name, age, result, confidence, image_path, analyzed_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            patient_name,
                            patient_age,
                            result["label_en"],
                            result["confidence"],
                            filepath.replace("\\", "/"),
                            session.get("username"),
                        ),
                    )
                    conn.commit()
                    patient_id = cursor.lastrowid
                finally:
                    cursor.close()
                    conn.close()

            # Stockage du résultat en session (évite les query strings)
            session["last_result"] = {
                "patient_name": patient_name,
                "patient_age" : patient_age,
                "image_path"  : filepath.replace("\\", "/"),
                "result"      : result,
                "analyzed_at" : datetime.now().strftime("%d/%m/%Y à %H:%M"),
            }

            return redirect(url_for("result"))

        except RuntimeError as e:
            flash(str(e), "danger")
            return redirect(url_for("predict"))
        except Exception as e:
            print(f"[ERREUR] predict(): {e}")
            flash("Une erreur est survenue lors de l'analyse.", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html")


# ── Résultat ─────────────────────────────────────────────────────────────────
@app.route("/result")
@login_required
def result():
    data = session.get("last_result")
    if not data:
        flash("Aucun résultat disponible. Lancez d'abord une analyse.", "info")
        return redirect(url_for("predict"))
    return render_template("result.html", data=data)


# ── Historique des patients ──────────────────────────────────────────────────
@app.route("/patients")
@login_required
def patients():
    conn = db_connect()
    records = []
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, name, age, result, confidence,
                       image_path, analyzed_by, created_at
                FROM patients
                ORDER BY created_at DESC
                """
            )
            records = cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    return render_template("patients.html", patients=records)


# ── Suppression d'un patient ─────────────────────────────────────────────────
@app.route("/patients/delete/<int:patient_id>", methods=["POST"])
@login_required
def delete_patient(patient_id):
    conn = db_connect()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # Récupérer le chemin de l'image pour la supprimer
            cursor.execute(
                "SELECT image_path FROM patients WHERE id = %s", (patient_id,)
            )
            row = cursor.fetchone()
            if row:
                # Supprimer l'image du disque si elle existe
                img_path = row["image_path"]
                if img_path and os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass
                # Supprimer l'enregistrement
                cursor.execute(
                    "DELETE FROM patients WHERE id = %s", (patient_id,)
                )
                conn.commit()
                flash("Enregistrement supprimé avec succès.", "success")
            else:
                flash("Enregistrement introuvable.", "warning")
        finally:
            cursor.close()
            conn.close()
    return redirect(url_for("patients"))


# ── À propos du modèle ───────────────────────────────────────────────────────
@app.route("/about")
@login_required
def about():
    return render_template("about.html")


# ── Déconnexion ──────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    username = session.get("username", "")
    session.clear()
    flash(f"Au revoir, {username} ! Vous avez été déconnecté.", "info")
    return redirect(url_for("login"))


# ── Initialisation de la base de données (route utilitaire) ─────────────────
@app.route("/init-db")
def initialize_db():
    """Route utilitaire pour initialiser la BDD (désactiver en production)."""
    try:
        init_db()
        flash("Base de données initialisée avec succès.", "success")
    except Exception as e:
        flash(f"Erreur d'initialisation : {e}", "danger")
    return redirect(url_for("login"))


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=app.config.get("DEBUG", False))
