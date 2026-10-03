# Méthodologie — Analyse honnête de l'EuroMillions

Ce document explique ce que le projet peut **réellement** faire, et pourquoi la
« prédiction » des numéros gagnants est impossible. Il accompagne le module
[`script/analyse_statistique.py`](../script/analyse_statistique.py) et le tableau
de bord [`rapport_honnete.html`](../rapport_honnete.html).

## 1. Pourquoi l'EuroMillions est imprévisible

L'EuroMillions est un **tirage aléatoire équitable** :

- Chaque tirage est **indépendant** des précédents (pas de mémoire).
- Toutes les combinaisons ont la **même probabilité**.
- La probabilité du jackpot est fixe : **1 sur 139 838 160**, soit
  `C(50,5) × C(12,2) = 2 118 760 × 66`.

Conséquence mathématique : **aucune** méthode — machine learning, réseaux de
neurones, « calcul quantique », analyse vidéo des boules, cycles lunaires,
suites de Fibonacci — ne peut faire mieux que le hasard pour deviner le prochain
tirage. L'espérance du nombre de bons numéros d'une grille est la même quelle
que soit la façon de la choisir :

```
E[bons numéros] = 5 × (5 / 50) = 0,5
E[bonnes étoiles] = 2 × (2 / 12) ≈ 0,33
```

### « Mais mon modèle a 80 % de précision sur l'historique ! »

C'est le signe d'un **surapprentissage** (overfitting) ou d'une **fuite de
données** (data leakage), pas d'un pouvoir prédictif. Un modèle qui mémorise le
passé n'a aucune valeur sur un tirage futur, par construction aléatoire. Le seul
test valable est une évaluation **walk-forward** stricte (ci-dessous), qui
ramène systématiquement toute stratégie au niveau du hasard.

## 2. Ce que le module calcule vraiment

### a) Statistiques descriptives
Fréquences des numéros et étoiles, écarts (gaps), distribution des sommes et de
la parité. Utile pour **décrire** l'historique, inutile pour **prédire**.

### b) Test d'équité (chi² d'uniformité)
On teste l'hypothèse H0 « chaque numéro a la même probabilité ». Le test porte
uniquement sur les **955 tirages du format actuel** (12 étoiles, depuis
septembre 2016), car le nombre d'étoiles a changé au fil du temps (9 → 11 → 12)
et mélanger les règles fausserait le résultat.

| Univers        | p-value | Verdict                         |
|----------------|---------|---------------------------------|
| Numéros (1–50) | 0,73    | compatible avec un tirage juste |
| Étoiles (1–12) | 0,16    | compatible avec un tirage juste |

Une p-value > 0,05 signifie que les écarts de fréquence observés sont du **bruit
statistique** : il n'existe pas de numéro réellement « chaud » ou « froid ».

### c) Évaluation honnête (backtest walk-forward)
Pour chaque tirage de test, la stratégie choisit 5 numéros **à partir du seul
passé** (aucune fuite). On mesure la moyenne de bons numéros :

| Stratégie            | Bons n° / tirage |
|----------------------|------------------|
| Aléatoire (référence)| ~0,47            |
| Numéros « chauds »   | ~0,48            |
| Numéros « froids »   | ~0,47            |
| Numéros « en retard »| ~0,48            |

Toutes convergent vers la valeur du hasard (~0,5). **Aucune n'a de pouvoir
prédictif.** C'est la preuve empirique, sur les données du projet.

## 3. La seule optimisation défendable : la valeur espérée

On ne peut pas changer la **probabilité de gagner**, mais on peut augmenter le
**gain conditionnel** (ce qu'on touche *si* on gagne). Le jackpot est partagé
entre tous les gagnants ; or les joueurs choisissent massivement :

- des **dates de naissance** (tous les numéros ≤ 31) ;
- des **suites** (1-2-3-4-5) et motifs réguliers (10-20-30-40-50) ;
- des grilles de **somme « moyenne »** autour de 128 ;
- des numéros **groupés** visuellement sur la grille.

En jouant une grille **peu populaire**, vous réduisez le risque de partage et
augmentez donc votre gain espéré — **sans** modifier votre chance de gagner.
C'est ce que fait `generate_ev_grids()` / le bouton du tableau de bord, via un
score de popularité (`popularity_score()`), identique en Python et en
JavaScript.

> Référence : plusieurs études (dont « Lottery Numbers and Ordered Statistics »
> et les travaux de Simon, Cox, Thaler & Ziemba) montrent que les joueurs ne
> choisissent pas uniformément, ce qui crée des combinaisons sur- et
> sous-jouées. Jouer les combinaisons sous-jouées améliore la valeur espérée.

## 4. Utilisation

```bash
# Rapport complet (test d'équité + backtest)
python3 script/analyse_statistique.py

# Sortie JSON (pour une API ou un tableau de bord)
python3 script/analyse_statistique.py --json

# Générer 5 grilles optimisées en valeur espérée
python3 script/analyse_statistique.py --generate 5

# Tests unitaires
python3 script/test_analyse_statistique.py
```

Le module n'utilise que la **bibliothèque standard Python** (aucun numpy, pandas,
torch…), il tourne donc partout, y compris sur le serveur de production.

## 5. Jeu responsable

Le loto est un divertissement, jamais un plan financier. L'espérance de gain
nette d'une grille EuroMillions est **négative**. Ne misez que ce que vous pouvez
perdre. Aide : **09 74 75 13 13** (Joueurs Info Service, France).
