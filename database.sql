-- ============================================================
-- database.sql — Initialisation de la base de données DermAI
-- ============================================================
-- Cours : Introduction à l'IA — ENSTAB | Dr. Amira Echtioui
-- Application : Diagnostic de Cancer Cutané (TD8)
-- ============================================================

-- Créer la base de données
CREATE DATABASE IF NOT EXISTS dermai_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE dermai_db;

-- ── Table users : gestion des comptes ──────────────────────────────────────
-- Les mots de passe sont hashés avec werkzeug.security (PBKDF2-SHA256)
CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(50)  NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL COMMENT 'Hash werkzeug (PBKDF2-SHA256)',
    created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Table patients : historique des analyses IA ────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL                    COMMENT 'Nom du patient',
    age         INT                                      COMMENT 'Âge du patient',
    result      VARCHAR(20)  NOT NULL                    COMMENT 'Malignant | Benign',
    confidence  FLOAT        NOT NULL                    COMMENT 'Confiance du modèle (%)',
    image_path  VARCHAR(255)                             COMMENT 'Chemin vers limage analysée',
    analyzed_by VARCHAR(50)                              COMMENT 'Médecin / utilisateur connecté',
    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP   COMMENT 'Date et heure de lanalyse'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Insertion de l'utilisateur admin par défaut ────────────────────────────
-- Mot de passe : admin1234 (hashé avec werkzeug generate_password_hash)
-- IMPORTANT : remplacer ce hash par le vôtre (voir database.py init_db())
INSERT IGNORE INTO users (username, password)
VALUES (
    'admin',
    'pbkdf2:sha256:260000$placeholder$changeme_run_python_init_db_instead'
);

-- ── Données de démonstration (optionnel) ───────────────────────────────────
-- INSERT INTO patients (name, age, result, confidence, analyzed_by)
-- VALUES
--     ('Patient Test A', 45, 'Benign',    92.5, 'admin'),
--     ('Patient Test B', 62, 'Malignant', 87.3, 'admin');

-- ── Vérification ─────────────────────────────────────────────────────────
SELECT 'Base de données dermai_db initialisée avec succès.' AS status;
SHOW TABLES;
