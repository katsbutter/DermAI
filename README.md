<div align="center">

# 🩺 DermAI

### Application Web de Diagnostic de Cancer Cutané par Intelligence Artificielle

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![MySQL](https://img.shields.io/badge/MySQL-8.0%2B-blue?logo=mysql&logoColor=white)](https://mysql.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

**Module :** Introduction à l'IA | **Établissement :** ENSTAB — Université de Carthage  
**Enseignante :** Dr. Amira Echtioui | **Année universitaire :** 2025/2026

</div>

---

## 📋 Table des matières

- [Présentation du projet](#-présentation-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du projet](#-architecture-du-projet)
- [Modèle IA — VGG16 Transfer Learning](#-modèle-ia--vgg16-transfer-learning)
- [Technologies utilisées](#-technologies-utilisées)
- [Installation et lancement](#-installation-et-lancement)
- [Utilisation](#-utilisation)
- [Structure de la base de données](#-structure-de-la-base-de-données)
- [Captures d'écran](#-captures-décran)
- [Concepts IA abordés](#-concepts-ia-abordés)
- [Avertissement](#️-avertissement)

---

## 🎯 Présentation du projet

**DermAI** est une application web médicale développée dans le cadre du **TD8 et TD9** du module *Introduction à l'IA*. Elle intègre un modèle de **Deep Learning** (VGG16 via Transfer Learning) pour classifier des images de lésions cutanées en **Bénin** ou **Malin**.

L'application permet à un professionnel de santé de :
- Se connecter de manière sécurisée
- Soumettre une image dermoscopique d'une lésion cutanée
- Obtenir instantanément une prédiction IA avec un indice de confiance
- Consulter et gérer l'historique complet des diagnostics (base MySQL)

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🔐 **Authentification sécurisée** | Connexion avec mot de passe hashé (PBKDF2-SHA256) |
| 🤖 **Analyse IA** | Classification VGG16 (Transfer Learning) avec score de confiance |
| 📊 **Tableau de bord** | Statistiques globales (total, malins, bénins) |
| 🖼️ **Upload drag & drop** | Prévisualisation de l'image avant soumission |
| 💾 **Persistance MySQL** | Historique complet des patients et diagnostics |
| 🔍 **Recherche & filtres** | Filtrage temps réel dans le tableau des patients |
| 🗑️ **Gestion des données** | Suppression sécurisée (BDD + fichier image) |
| 📱 **Interface responsive** | Compatible desktop, tablette et mobile |
| 📖 **Page "À propos"** | Documentation du modèle IA (architecture, dataset, métriques) |

---

## 📁 Architecture du projet

```
DermAI/
│
├── 📄 app.py                    # Application Flask — routes et logique principale
├── 📄 config.py                 # Configuration centralisée (BDD, modèle, upload)
├── 📄 database.py               # Connexion MySQL et initialisation des tables
├── 📄 database.sql              # Script SQL de création des tables
│
├── 📁 model/
│   └── vgg16_skin_cancer.h5    # Modèle Keras pré-entraîné (~130 Mo, non versionné)
│
├── 📁 scripts/
│   └── train_model.py          # Script d'entraînement VGG16 (Transfer Learning)
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── main.css            # Feuille de style principale
│   ├── 📁 js/
│   │   └── main.js             # JavaScript (sidebar, horloge, upload)
│   ├── 📁 images/              # Captures, courbes d'apprentissage, confusion matrix
│   └── 📁 uploads/             # Images soumises par les utilisateurs
│
├── 📁 templates/
│   ├── base.html               # Template de base (sidebar + topbar)
│   ├── login.html              # Page de connexion
│   ├── dashboard.html          # Tableau de bord avec statistiques
│   ├── predict.html            # Formulaire d'analyse
│   ├── result.html             # Résultat de la prédiction
│   ├── patients.html           # Historique des patients
│   └── about.html              # Documentation du modèle IA
│
├── 📄 requirements.txt         # Dépendances Python
├── 📄 .env.example             # Template des variables d'environnement
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🧠 Modèle IA — VGG16 Transfer Learning

### Architecture

```
Entrée image 224×224×3
        ↓
VGG16 (couches gelées — poids ImageNet)
        ↓
Flatten
        ↓
Dropout(0.5)
        ↓
Dense(256, activation='relu')
        ↓
Dense(1, activation='sigmoid')
        ↓
Sortie : probabilité de malignité [0, 1]
  → > 0.5 : Malin | ≤ 0.5 : Bénin
```

### Paramètres d'entraînement

| Paramètre | Valeur |
|---|---|
| Modèle de base | VGG16 (ImageNet) |
| Optimiseur | Adam (lr = 1e-4) |
| Fonction de perte | Binary Cross-Entropy |
| Epochs | 10 |
| Batch size | 32 |
| Seuil de décision | 0.5 (sigmoid) |
| Prétraitement | Normalisation pixels ÷ 255 → [0, 1] |
| Augmentation | Rotation, zoom, flip, shear |

### Dataset

Le modèle est entraîné sur le dataset **Skin Cancer Malignant vs Benign** (Kaggle) :
- ~2 637 images d'entraînement
- ~660 images de test
- 2 classes : `benign` (0) / `malignant` (1)

> **Pour entraîner le modèle vous-même :**
> ```bash
> # Télécharger le dataset
> kaggle datasets download -d fanconic/skin-cancer-malignant-vs-benign
> unzip skin-cancer-malignant-vs-benign.zip -d data/
>
> # Lancer l'entraînement
> python scripts/train_model.py
> ```

---

## 🛠️ Technologies utilisées

| Couche | Technologie | Rôle |
|---|---|---|
| Backend | Flask 2.3+ | Framework web Python |
| IA | TensorFlow / Keras | Chargement et inférence VGG16 |
| Base de données | MySQL 8.0 | Persistance des données patients |
| Sécurité | Werkzeug | Hachage des mots de passe |
| Frontend | Bootstrap 5.3 | Interface responsive |
| Icônes | Bootstrap Icons | Iconographie |
| Polices | Google Fonts (Inter) | Typographie |

---

## 🚀 Installation et lancement

### Prérequis

- Python 3.9 ou supérieur
- MySQL 8.0 (via XAMPP, WAMP ou installation native)
- pip

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/VOTRE-NOM/DermAI.git
cd DermAI
```

### Étape 2 — Créer un environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 — Configurer les variables d'environnement

```bash
cp .env.example .env
```

Editez le fichier `.env` :

```env
SECRET_KEY=votre_cle_secrete
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=dermai_db
FLASK_DEBUG=False
```

### Étape 5 — Initialiser la base de données

```bash
# Démarrez MySQL (XAMPP, WAMP ou service système)
# Puis lancez l'application une première fois et accédez à :
# http://127.0.0.1:5000/init-db
```

Ou depuis MySQL directement :
```sql
source database.sql;
```

> **Note :** Le mot de passe admin sera `admin1234`. Pour le changer, modifiez `database.py` → `init_db()`.

### Étape 6 — Ajouter le modèle IA

Placez votre fichier `vgg16_skin_cancer.h5` dans le dossier `model/`.

> Le modèle n'est pas inclus dans le dépôt (taille ~130 Mo).  
> Entraînez-le avec `python scripts/train_model.py` ou obtenez-le séparément.

### Étape 7 — Lancer l'application

```bash
python app.py
```

Ouvrez votre navigateur : **http://127.0.0.1:5000**

---

## 📖 Utilisation

### Connexion

| Identifiant | Mot de passe |
|---|---|
| `admin` | `admin1234` |

### Workflow typique

```
1. Se connecter → /login
2. Tableau de bord → /dashboard   (statistiques globales)
3. Nouvelle analyse → /predict    (upload image + infos patient)
4. Voir le résultat → /result     (diagnostic + indice de confiance)
5. Historique → /patients         (tous les diagnostics enregistrés)
```

### Routes disponibles

| Route | Méthode | Description | Auth |
|---|---|---|---|
| `/` | GET | Redirection automatique | — |
| `/login` | GET, POST | Authentification | Non |
| `/dashboard` | GET | Tableau de bord | ✓ |
| `/predict` | GET, POST | Analyse IA | ✓ |
| `/result` | GET | Résultat de l'analyse | ✓ |
| `/patients` | GET | Historique des patients | ✓ |
| `/patients/delete/<id>` | POST | Supprimer un patient | ✓ |
| `/about` | GET | Documentation du modèle | ✓ |
| `/logout` | GET | Déconnexion | — |
| `/init-db` | GET | Initialiser la BDD (setup) | — |

---

## 🗄️ Structure de la base de données

```sql
-- Table users : comptes utilisateurs
CREATE TABLE users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(50)  NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL,  -- Hash PBKDF2-SHA256
    created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- Table patients : historique des analyses
CREATE TABLE patients (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    age         INT,
    result      VARCHAR(20)  NOT NULL,  -- 'Malignant' | 'Benign'
    confidence  FLOAT        NOT NULL,  -- Probabilité en %
    image_path  VARCHAR(255),
    analyzed_by VARCHAR(50),
    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📸 Captures d'écran

> *(Remplacez les emplacements ci-dessous par vos captures réelles)*

### Page de connexion
![Login](static/images/screenshot_login.png)

### Tableau de bord
![Dashboard](static/images/screenshot_dashboard.png)

### Analyse IA
![Predict](static/images/screenshot_predict.png)

### Résultat de la prédiction
![Result](static/images/screenshot_result.png)

### Historique des patients
![Patients](static/images/screenshot_patients.png)

---

## 📚 Concepts IA abordés

Ce projet mobilise les notions étudiées tout au long du cours :

| TD | Concept | Application dans DermAI |
|---|---|---|
| TD1 | Types d'apprentissage | Classification supervisée (Bénin / Malin) |
| TD2 | Arbre de décision | Compréhension de la classification binaire |
| TD3 | SVM | Notion de frontière de décision |
| TD5 | Réseaux de neurones (MLP) | Couches Dense, fonctions d'activation |
| TD6 | CNN | Couches Conv2D de VGG16 |
| TD7 | Transfer Learning | VGG16 + ImageNet + couches personnalisées |
| TD8 | Application Web IA | Flask + MySQL + interface utilisateur |

### Pipeline de prédiction complet

```
Image utilisateur
      ↓
Redimensionnement 224×224 px
      ↓
Normalisation : pixels ÷ 255 → [0, 1]
      ↓
Expansion dimensions : (224,224,3) → (1,224,224,3)
      ↓
VGG16 : extraction de features (couches gelées)
      ↓
Flatten → Dropout(0.5) → Dense(256, ReLU)
      ↓
Dense(1, Sigmoid) → valeur entre 0 et 1
      ↓
Décision : > 0.5 → Malin | ≤ 0.5 → Bénin
      ↓
Enregistrement MySQL + affichage résultat
```

---

## ⚠️ Avertissement

> Cette application est développée **à des fins éducatives et pédagogiques** dans le cadre du module *Introduction à l'IA* de l'ENSTAB.
>
> Elle **ne constitue en aucun cas un dispositif médical** et **ne remplace pas** l'avis d'un dermatologue qualifié. Tout diagnostic de cancer cutané doit être réalisé par un professionnel de santé.

---

## 👩‍🏫 Encadrement académique

**Module :** Introduction à l'Intelligence Artificielle  
**Enseignante :** Dr. Amira Echtioui — Maître assistante à l'ENSTAB  
**Contact :** amira.echtioui@enstab.ucar.tn  
**Établissement :** ENSTAB — Université de Carthage  
**Année :** 2025/2026

---

## 📄 Licence

Ce projet est sous licence [MIT](LICENSE).
