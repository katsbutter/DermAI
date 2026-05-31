"""
database.py — Gestion de la connexion MySQL
============================================
Fournit deux fonctions :
  - db_connect() : retourne une connexion MySQL active
  - init_db()    : crée les tables si elles n'existent pas

Schéma de base de données (conforme TD8) :
  - users    : gestion des comptes utilisateurs (mot de passe hashé)
  - patients : historique des analyses IA

Cours : Introduction à l'IA — ENSTAB | Dr. Amira Echtioui
"""

import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash

from config import Config


def db_connect():
    """
    Établit et retourne une connexion MySQL.
    Retourne None en cas d'échec (l'application reste fonctionnelle).
    """
    try:
        conn = mysql.connector.connect(
            host     = Config.DB_HOST,
            user     = Config.DB_USER,
            password = Config.DB_PASSWORD,
            database = Config.DB_NAME,
        )
        return conn
    except Error as e:
        print(f"[DB] Erreur de connexion MySQL : {e}")
        return None


def init_db():
    """
    Initialise la base de données :
    1. Crée la base 'dermai_db' si elle n'existe pas
    2. Crée les tables users et patients
    3. Insère l'utilisateur admin par défaut (mot de passe hashé)
    """
    try:
        # Connexion sans sélectionner de base de données
        conn = mysql.connector.connect(
            host     = Config.DB_HOST,
            user     = Config.DB_USER,
            password = Config.DB_PASSWORD,
        )
        cursor = conn.cursor()

        # ── Création de la base de données ─────────────────────────────────
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {Config.DB_NAME} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.execute(f"USE {Config.DB_NAME}")

        # ── Table users ─────────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                username   VARCHAR(50)  NOT NULL UNIQUE,
                password   VARCHAR(255) NOT NULL COMMENT 'Mot de passe hashé (werkzeug)',
                created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # ── Table patients ──────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                name        VARCHAR(100) NOT NULL,
                age         INT,
                result      VARCHAR(20)  NOT NULL COMMENT 'Malignant | Benign',
                confidence  FLOAT        NOT NULL COMMENT 'Probabilité en pourcentage',
                image_path  VARCHAR(255),
                analyzed_by VARCHAR(50)  COMMENT 'Nom de lutilisateur connecté',
                created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # ── Utilisateur admin par défaut ────────────────────────────────────
        # Mot de passe hashé avec werkzeug (jamais en clair)
        cursor.execute(
            "SELECT COUNT(*) FROM users WHERE username = 'admin'"
        )
        (count,) = cursor.fetchone()
        if count == 0:
            hashed_pw = generate_password_hash("admin1234")
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                ("admin", hashed_pw),
            )
            print("[DB] Utilisateur 'admin' créé (mot de passe : admin1234)")

        conn.commit()
        print("[DB] Base de données initialisée avec succès.")

    except Error as e:
        print(f"[DB] Erreur lors de l'initialisation : {e}")
        raise
    finally:
        if "cursor" in dir():
            cursor.close()
        if "conn" in dir() and conn.is_connected():
            conn.close()
