"""
Projet Complet d'Analyse de Données avec Iris Dataset

Ce script effectue une analyse complète du dataset Iris, incluant:
1. Chargement et inspection des données
2. Nettoyage et préparation des données
3. Analyse statistique de base
4. Regroupement et agrégation des données
5. Créations de visualisations (graphiques linéaires, en barres, histogrammes, et dispersions)
"""

# ============================================================================
# SECTION 1 : IMPORTER LES BIBLIOTHÈQUES NÉCESSAIRES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration pour les visualisations
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 7)

print("✓ Bibliothèques importées avec succès\n")


# ============================================================================
# SECTION 2 : CHARGER ET INSPECTER LE DATASET
# ============================================================================

print("="*80)
print("SECTION 1 : CHARGER ET INSPECTER LE DATASET")
print("="*80)

try:
    # Charger les données Iris depuis sklearn
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    # Ajouter une colonne de date pour la visualisation temporelle
    # (simule des enregistrements sur plusieurs mois)
    dates = pd.date_range(start='2023-01-01', periods=len(df), freq='2D')
    df['date'] = dates
    
    print(f"\n✓ Dataset chargé avec succès!")
    print(f"  Nombre de lignes: {df.shape[0]}")
    print(f"  Nombre de colonnes: {df.shape[1]}\n")
    
except Exception as e:
    print(f"✗ Erreur lors du chargement du dataset: {e}")
    exit()

# Afficher les premières lignes du dataset
print("PREMIÈRES LIGNES DU DATASET (using .head()):")
print("-" * 80)
print(df.head(10))
print(f"\n... Total de {len(df)} lignes ...\n")

# Explorer la structure du dataset
print("STRUCTURE ET TYPES DE DONNÉES:")
print("-" * 80)
print("\nTypes de données (.dtypes):")
print(df.dtypes)
print("\nDimensions du dataset:")
print(f"  Lignes: {df.shape[0]}")
print(f"  Colonnes: {df.shape[1]}\n")

# Identifier les valeurs manquantes
print("ANALYSE DES VALEURS MANQUANTES:")
print("-" * 80)
missing_values = df.isnull().sum()
print("\nValeurs manquantes par colonne:")
print(missing_values)
print(f"\nTotal des valeurs manquantes: {missing_values.sum()}")

if missing_values.sum() == 0:
    print("✓ Aucune valeur manquante détectée!\n")


# ============================================================================
# SECTION 3 : NETTOYER ET PRÉPARER LES DONNÉES
# ============================================================================

print("\n" + "="*80)
print("SECTION 2 : NETTOYER ET PRÉPARER LES DONNÉES")
print("="*80)

# Vérifier les doublons
print(f"\nNombre de doublons: {df.duplicated().sum()}")

if df.duplicated().sum() > 0:
    df_clean = df.drop_duplicates()
    print(f"✓ {df.duplicated().sum()} doublons supprimés")
else:
    df_clean = df.copy()
    print("✓ Aucun doublon trouvé")

# Vérifier les valeurs manquantes et les remplir si nécessaire
if df_clean.isnull().sum().sum() > 0:
    print("\nRemplissage des valeurs manquantes...")
    # Remplir les valeurs numériques avec la médiane
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    print("✓ Valeurs manquantes remplies avec la médiane")
else:
    print("✓ Aucune valeur manquante à remplir")

print(f"\nDataset prêt pour l'analyse: {df_clean.shape[0]} lignes, {df_clean.shape[1]} colonnes")


# ============================================================================
# SECTION 4 : ANALYSE STATISTIQUE DE BASE
# ============================================================================

print("\n" + "="*80)
print("SECTION 3 : ANALYSE STATISTIQUE DE BASE")
print("="*80)

# Statistiques descriptives
print("\nSTATISTIQUES DESCRIPTIVES (.describe()):")
print("-" * 80)
statistics = df_clean.describe()
print(statistics)

# Interprétation des statistiques
print("\nINTERPRÉTATION DES STATISTIQUES:")
print("-" * 80)
for col in df_clean.select_dtypes(include=[np.number]).columns:
    data = df_clean[col]
    print(f"\n{col}:")
    print(f"  Moyenne: {data.mean():.2f}")
    print(f"  Médiane: {data.median():.2f}")
    print(f"  Écart-type: {data.std():.2f}")
    print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")
    print(f"  Plage (IQR): Q1={data.quantile(0.25):.2f}, Q3={data.quantile(0.75):.2f}")


# ============================================================================
# SECTION 5 : REGROUPEMENT ET AGRÉGATION DES DONNÉES
# ============================================================================

print("\n" + "="*80)
print("SECTION 4 : REGROUPEMENT ET AGRÉGATION DES DONNÉES")
print("="*80)

# Regrouper par espèce et calculer les moyennes
print("\nMOYENNES PAR ESPÈCE (using .groupby() et .mean()):")
print("-" * 80)
species_means = df_clean.groupby('species').mean()
print(species_means)

# Nombre de observations par espèce
print("\nNOMBRE D'OBSERVATIONS PAR ESPÈCE:")
print("-" * 80)
species_counts = df_clean['species'].value_counts()
print(species_counts)

# Regroupement avec plusieurs statistiques
print("\nSTATISTIQUES AVANCÉES PAR ESPÈCE:")
print("-" * 80)
species_stats = df_clean.groupby('species')[df_clean.select_dtypes(include=[np.number]).columns].agg(['mean', 'std', 'min', 'max'])
print(species_stats)

# Motifs et résultats intéressants
print("\nPATRONS ET RÉSULTATS INTÉRESSANTS:")
print("-" * 80)
print("\n1. Distribution des espèces:")
for species, count in species_counts.items():
    percentage = (count / len(df_clean)) * 100
    print(f"   {species}: {count} observations ({percentage:.1f}%)")

print("\n2. Comparaison des longueurs de pétales par espèce:")
for species in df_clean['species'].unique():
    petal_length = df_clean[df_clean['species'] == species]['petal length (cm)'].mean()
    print(f"   {species}: {petal_length:.2f} cm (moyenne)")

print("\n3. Espèce avec le plus grande longueur sépale moyenne:")
sepal_means = df_clean.groupby('species')['sepal length (cm)'].mean()
print(f"   {sepal_means.idxmax()}: {sepal_means.max():.2f} cm")


# ============================================================================
# SECTION 6 : VISUALISATIONS
# ============================================================================

print("\n" + "="*80)
print("SECTION 5 : CRÉER DES VISUALISATIONS")
print("="*80)

# Créer une figure avec 4 sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ----
# GRAPHIQUE 1 : GRAPHIQUE LINÉAIRE - Tendance temporelle
# ----
print("\n1. Création du graphique linéaire (tendance temporelle)...")
ax1 = axes[0, 0]

# Calculer la moyenne mobile de la longueur des sépales par jour
daily_avg = df_clean.groupby('date')['sepal length (cm)'].mean()
ax1.plot(daily_avg.index, daily_avg.values, linewidth=2.5, marker='o', markersize=4, color='#2E86AB', label='Moyenne mobile')
ax1.fill_between(daily_avg.index, daily_avg.values, alpha=0.3, color='#2E86AB')

ax1.set_title('Tendance de la Longueur des Sépales au Fil du Temps', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Longueur des Sépales (cm)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# ----
# GRAPHIQUE 2 : GRAPHIQUE EN BARRES - Moyenne par espèce
# ----
print("2. Création du graphique en barres (comparaison par catégories)...")
ax2 = axes[0, 1]

species_petal = df_clean.groupby('species')['petal length (cm)'].mean()
colors = ['#A23B72', '#F18F01', '#C73E1D']
bars = ax2.bar(species_petal.index, species_petal.values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# Ajouter les valeurs sur les barres
for bar, value in zip(bars, species_petal.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_title('Longueur Moyenne des Pétales par Espèce', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Espèce', fontsize=12, fontweight='bold')
ax2.set_ylabel('Longueur des Pétales (cm)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ----
# GRAPHIQUE 3 : HISTOGRAMME - Distribution
# ----
print("3. Création de l'histogramme (distribution)...")
ax3 = axes[1, 0]

data_to_plot = df_clean['sepal length (cm)']
n, bins, patches = ax3.hist(data_to_plot, bins=20, edgecolor='black', linewidth=1.5, color='#06A77D', alpha=0.7)

# Ajouter une ligne de moyenne
mean_val = data_to_plot.mean()
ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Moyenne: {mean_val:.2f} cm')

ax3.set_title('Distribution de la Longueur des Sépales', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('Longueur des Sépales (cm)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Fréquence', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# ----
# GRAPHIQUE 4 : SCATTER PLOT - Relation entre deux variables
# ----
print("4. Création du nuage de points (relation entre variables)...")
ax4 = axes[1, 1]

# Utiliser des couleurs différentes pour chaque espèce
species_colors = {'setosa': '#E63946', 'versicolor': '#F1FAEE', 'virginica': '#A8DADC'}
for species in df_clean['species'].unique():
    species_data = df_clean[df_clean['species'] == species]
    ax4.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
               label=species, s=100, alpha=0.7, edgecolors='black', linewidth=1)

# Ajouter une ligne de tendance
z = np.polyfit(df_clean['sepal length (cm)'], df_clean['petal length (cm)'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_clean['sepal length (cm)'].min(), df_clean['sepal length (cm)'].max(), 100)
ax4.plot(x_trend, p(x_trend), "k--", linewidth=2, label='Tendance', alpha=0.7)

ax4.set_title('Relation entre Longueur des Sépales et Longueur des Pétales', fontsize=14, fontweight='bold', pad=20)
ax4.set_xlabel('Longueur des Sépales (cm)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Longueur des Pétales (cm)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/benjamin/Documents/PLP ACADAMY/1/py_week7/visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualisations sauvegardées dans 'visualizations.png'\n")
plt.close(fig)


# ============================================================================
# VISUALISATIONS ADDITIONNELLES
# ============================================================================

# Heatmap de corrélation
fig2, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
correlation_matrix = df_clean[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Corrélation'}, 
            fmt='.2f', linewidths=1, linecolor='black')
ax.set_title('Matrice de Corrélation des Variables Numériques', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/benjamin/Documents/PLP ACADAMY/1/py_week7/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# Boîte à moustaches (boxplot) par espèce
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
# Exclure la colonne 'date'
numeric_cols = [col for col in numeric_cols if col != 'date']

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    sns.boxplot(data=df_clean, x='species', y=col, ax=ax, palette='Set2')
    ax.set_title(f'Distribution de {col} par Espèce', fontsize=12, fontweight='bold')
    ax.set_xlabel('Espèce', fontsize=11, fontweight='bold')
    ax.set_ylabel(col, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/benjamin/Documents/PLP ACADAMY/1/py_week7/boxplots.png', dpi=300, bbox_inches='tight')
plt.close(fig3)


# ============================================================================
# RÉSUMÉ
# ============================================================================

print("Analyse et visualisations générées avec succès. Fichiers: visualizations.png, correlation_heatmap.png, boxplots.png")
