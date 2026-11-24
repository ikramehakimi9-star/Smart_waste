# SmartWaste : classification d'images pour le tri des déchets

Projet de machine learning classique (sans réseau de neurones) qui reconnaît le type d'un déchet à partir d'une photo.
Chaque image est décrite par sa **forme** (HOG) et sa **couleur** (histogrammes HSV), puis classée par un **SVM**.

**Classes :** `cardboard` · `glass` · `metal` · `paper` · `plastic` · `trash`

## Résultats

Évaluation finale sur un jeu de **test jamais utilisé** pour l'entraînement ni pour les réglages (379 images) :

| Jeu | Accuracy | F1-macro |
|---|---|---|
| Validation | 0,683 | 0,676 |
| **Test** | **0,633** | **0,613** |

Pour comparaison, répondre au hasard avec 6 classes donne environ 17 %.

F1 par classe sur le test :

| cardboard | glass | metal | paper | plastic | trash |
|---|---|---|---|---|---|
| 0,713 | 0,556 | 0,525 | 0,763 | 0,604 | 0,513 |

Le papier et le carton sont les mieux reconnus. Le verre, le métal et le plastique se confondent entre eux (matériaux brillants ou transparents, mal décrits par la forme et la couleur seules).

## Données

- 2527 images au départ, **2524** après nettoyage (0 image corrompue, 3 doublons exacts supprimés).
- Un dossier par classe ; le nom du dossier sert d'étiquette.
- Classes déséquilibrées : `paper` 594 images, `trash` 137.
- Source du dataset : kaggle.



## Méthode

1. **Nettoyage** : images illisibles ou trop petites, doublons (empreinte MD5).
2. **Séparation stratifiée** : 70 % entraînement, 15 % validation, 15 % test, faite *avant* toute extraction pour éviter les fuites de données.
3. **Extraction des caractéristiques** (1796 valeurs par image, redimensionnée en 128 × 128) :
   - forme : HOG sur l'image en gris améliorée par CLAHE (1764 valeurs) ;
   - couleur : histogrammes H, S et V normalisés (16 + 8 + 8 = 32 valeurs).
4. **Augmentation** du jeu d'entraînement uniquement : miroir, rotation de 10°, image plus lumineuse (7064 lignes).
5. **Standardisation** (`StandardScaler`) puis **PCA** à 150 composantes (75,2 % de la variance conservée), ajustées sur le train seulement.
6. **SVM** à noyau RBF avec `class_weight="balanced"`. Réglages C et gamma choisis par une petite recherche (6 essais) sur la validation : C = 1, gamma = `scale`.
7. **Évaluation** : accuracy, F1-macro, rapport par classe, matrices de confusion, images mal classées.
8. **Sauvegarde** du scaler, de la PCA, du modèle et de tous les paramètres de préparation dans un seul fichier `.joblib`.

## Structure du dépôt

```
.
├── tri_dechets_hog_couleur_svm_rapide.ipynb   # notebook complet, de bout en bout
├── dataset/                                   # images (non incluses)
│   ├── cardboard/ glass/ metal/ paper/ plastic/ trash/
├── output/                                    # généré par le notebook
│   ├── modele_hog_couleur_svm.joblib
│   └── figures (.png)
└── README.md
```

## Installation

Python 3.9 ou plus récent est conseillé.

```bash
pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib seaborn joblib tqdm jupyter
```

## Utilisation

1. Ouvre le notebook et modifie `BASE_DIR` dans la partie « Paramètres » pour qu'il pointe vers ton dossier de projet.
2. Exécute les cellules dans l'ordre (l'entraînement final prend environ une minute).
3. Pour tester une nouvelle image, place-la dans le dossier du projet et règle `TEST_IMAGE`.

### Réutiliser le modèle sauvegardé

```python
import joblib

paquet = joblib.load("output/modele_hog_couleur_svm.joblib")
scaler, pca, model = paquet["scaler"], paquet["pca"], paquet["model"]
classes = paquet["classes"]
```

Une nouvelle image doit subir **exactement** la même préparation qu'à l'entraînement : redimensionnement 128 × 128, niveaux de gris, CLAHE, HOG, histogrammes HSV, concaténation, puis `scaler.transform` et `pca.transform`.
Attention : les réglages de CLAHE ne sont pas dans le fichier sauvegardé, il faut les recréer à l'identique :

```python
import cv2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

## Limites

- Score modeste : HOG et couleur décrivent mal les reflets et la transparence.
- Jeu de test de taille modeste (379 images) : les scores par classe, surtout pour `trash` (21 images), restent bruités.
- Recherche d'hyperparamètres volontairement réduite pour rester rapide.
- Seuls les doublons exacts sont détectés.
- Les photos du dataset sont propres (fond uni, objet centré) ; des photos prises « dans la vraie vie » seraient plus difficiles.

## Pistes d'amélioration

- Réseau de neurones convolutif ou transfert d'apprentissage.
- Plus de données et d'augmentations.
- Recherche d'hyperparamètres plus large.
- API Flask qui expose le modèle (envoi d'une image, retour de la classe et de la confiance).

## Auteur

Auteur:Hakimi Ikram, (EST Salé).
