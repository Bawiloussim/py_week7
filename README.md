# 📊 Projet Complet d'Analyse de Données - Dataset Iris

## 📌 Vue d'ensemble

Ce projet contient une **analyse complète et professionnelle** du dataset Iris en utilisant Python, Pandas, Matplotlib et Seaborn. Le projet répond à tous les critères demandés et fournit des insights détaillés sur les données.

---

## 🎯 Objectifs Réalisés

### ✅ Tâche 1 : Chargement et Nettoyage des Données
- ✓ Chargement du dataset Iris (150 observations, 4 variables numériques + 1 catégorie)
- ✓ Inspection avec `.head()` - affiche les 10 premières lignes
- ✓ Vérification des types de données (`.dtypes`)
- ✓ Détection des valeurs manquantes - **Aucune trouvée** ✓
- ✓ Vérification des doublons - **Aucun trouvé** ✓
- ✓ Dataset prêt pour l'analyse

### ✅ Tâche 2 : Analyse Statistique de Base
- ✓ **Statistiques descriptives** avec `.describe()`:
  - Longueur sépale: 5.84 cm ± 0.83 cm
  - Largeur sépale: 3.06 cm ± 0.44 cm
  - Longueur pétale: 3.76 cm ± 1.77 cm
  - Largeur pétale: 1.20 cm ± 0.76 cm

- ✓ **Regroupement par espèce** avec `.groupby()`:
  - Setosa: petites fleurs (longueur pétale: 1.46 cm)
  - Versicolor: fleurs moyennes (longueur pétale: 4.26 cm)
  - Virginica: grandes fleurs (longueur pétale: 5.55 cm)

- ✓ **Patterns identifiés**:
  - Dataset équilibré (50 observations par espèce)
  - Forte corrélation entre longueur et largeur des pétales
  - Progression claire de la taille des fleurs par espèce

### ✅ Tâche 3 : Visualisations (4+ graphiques)

#### 1. **Graphique Linéaire** 📈
- Tendance temporelle de la longueur des sépales
- Affiche l'évolution sur plusieurs mois
- Inclut une zone d'intensité sous la courbe

#### 2. **Graphique en Barres** 📊
- Comparaison de la longueur moyenne des pétales par espèce
- Codes couleurs distincts pour chaque espèce
- Valeurs numériques affichées sur les barres

#### 3. **Histogramme** 📉
- Distribution de la longueur des sépales
- 20 bacs pour une meilleure granularité
- Ligne de moyenne affichée en rouge

#### 4. **Scatter Plot** 🔵
- Relation entre longueur et largeur des sépales
- Couleurs différentes par espèce
- Ligne de tendance polynomiale incluse

#### 5. **Bonus : Heatmap de Corrélation** 🔥
- Matrice de corrélation entre toutes les variables numériques
- Affiche les valeurs de corrélation
- Échelle de couleur pour visualiser les relations

#### 6. **Bonus : Boîtes à Moustaches** 📦
- Distribution par variable et par espèce
- 4 graphiques montrant les statistiques quartiles
- Comparaison visuelle entre espèces

---

## 📁 Structure des Fichiers

```
py_week7/
├── data_analysis.py              # Script principal d'analyse
├── visualizations.png            # 4 graphiques principaux
├── correlation_heatmap.png       # Matrice de corrélation
├── boxplots.png                  # Distributions en boîtes à moustaches
└── README.md                     # Ce fichier
```

---

## 🚀 Exécution du Programme

### Prérequis
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Lancer l'analyse
```bash
python data_analysis.py
```

**Résultat** : 
- Affichage détaillé des analyses dans la console
- Génération de 3 fichiers PNG avec les visualisations

---

## 📊 Résultats Clés

### Données Générales
| Métrique | Valeur |
|----------|--------|
| **Total d'observations** | 150 |
| **Variables** | 6 (4 numériques + 1 catégorique + 1 date) |
| **Valeurs manquantes** | 0 |
| **Doublons** | 0 |

### Distribution par Espèce
| Espèce | Observations | % |
|--------|-------------|---|
| **Setosa** | 50 | 33.3% |
| **Versicolor** | 50 | 33.3% |
| **Virginica** | 50 | 33.3% |

### Longueurs Moyennes par Espèce
| Espèce | Sépale (cm) | Pétale (cm) |
|--------|-----------|-----------|
| **Setosa** | 5.01 | 1.46 |
| **Versicolor** | 5.94 | 4.26 |
| **Virginica** | 6.59 | 5.55 |

---

## 🔍 Insights et Conclusions

1. **Parfaite Balance** : Le dataset est parfaitement équilibré avec 50 observations pour chaque espèce.

2. **Distinction Claire** : Les trois espèces montrent des patterns distincts, particulièrement pour la longueur des pétales.

3. **Corrélation Forte** : La longueur et la largeur des pétales sont fortement corrélées (utile pour la classification).

4. **Données de Qualité** : Aucune valeur manquante ou doublon - prêt pour l'analyse.

5. **Progression Graduelle** : Setosa < Versicolor < Virginica en termes de taille des fleurs.

---

## 🛠️ Technologies Utilisées

| Outil | Version | Utilité |
|-------|---------|---------|
| **Python** | 3.13.7 | Langage de programmation |
| **Pandas** | Latest | Manipulation et analyse de données |
| **Matplotlib** | Latest | Création de graphiques |
| **Seaborn** | Latest | Visualisations statistiques avancées |
| **Scikit-learn** | Latest | Chargement du dataset Iris |
| **NumPy** | Latest | Calculs numériques |

---

## 📝 Gestion des Erreurs

Le script inclut la gestion complète des erreurs :

```python
try:
    # Chargement du dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print("✓ Dataset chargé avec succès!")
except Exception as e:
    print(f"✗ Erreur lors du chargement du dataset: {e}")
    exit()
```

---

## 💡 Personnalisations Effectuées

✨ **Au-delà des critères demandés** :
- ✅ Ajout d'une colonne de date pour la visualisation temporelle
- ✅ Utilisation avancée de Seaborn pour un style professionnel
- ✅ Génération de plusieurs fichiers PNG
- ✅ Inclusion d'une heatmap de corrélation
- ✅ Création de boîtes à moustaches pour chaque variable
- ✅ Interprétation détaillée de chaque statistique
- ✅ Motifs et insights d'affaires (business insights)
- ✅ Console output colorisée et structurée

---

## 📚 Apprentissages Clés

Ce projet démontre :

1. **Pandas Mastery** : `.head()`, `.dtypes`, `.isnull()`, `.groupby()`, `.mean()`, `.describe()`
2. **Matplotlib Skills** : Création de graphiques linéaires, en barres, histogrammes, et dispersions
3. **Seaborn Expertise** : Heatmaps, boîtes à moustaches, et styles avancés
4. **Data Cleaning** : Gestion des valeurs manquantes et doublons
5. **Statistical Analysis** : Calcul et interprétation des statistiques
6. **Data Visualization** : Communication efficace des insights

---

## 🎓 Utilisation Pédagogique

Ce projet est idéal pour apprendre :
- ✓ Les bases de Pandas pour le nettoyage de données
- ✓ La création de visualisations professionnelles
- ✓ L'analyse exploratoire de données (EDA)
- ✓ Les bonnes pratiques en matière de gestion des erreurs
- ✓ La documentation du code et des résultats

---

## 👤 Auteur

Projet d'analyse de données - PLP Academy Week 7

---

## 📄 Licence

Ce projet est fourni à titre éducatif.

---

## 🙏 Remerciements

- Dataset Iris fourni par Scikit-learn
- Pandas pour la manipulation de données
- Matplotlib et Seaborn pour les visualisations
- PLP Academy pour le framework du cours

---

**Dernière mise à jour** : 10 novembre 2025
**Status** : ✅ Complet et testé
# py_week7
