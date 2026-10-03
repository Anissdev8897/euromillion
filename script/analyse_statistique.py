#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse statistique honnête de l'EuroMillions.
====================================================================

AVERTISSEMENT SCIENTIFIQUE
--------------------------
L'EuroMillions est un tirage *aléatoire équitable*. Chaque tirage est
indépendant des précédents et toutes les combinaisons possibles ont
*exactement* la même probabilité de sortir. Il est donc mathématiquement
impossible de prédire les numéros gagnants mieux que le pur hasard — ni par
machine learning, ni par analyse vidéo, ni par cycles lunaires, ni par
« calcul quantique ». Toute « précision » élevée affichée par un prédicteur
sur l'historique est un artefact de surapprentissage (overfitting), pas une
capacité réelle de prédiction.

CE QUE CE MODULE FOURNIT (de façon rigoureuse et vérifiable)
------------------------------------------------------------
1. Statistiques descriptives : fréquences, écarts (gaps), sommes, parité.
2. Test du chi² d'uniformité : vérifie quantitativement que le tirage est
   équitable (aucun numéro n'est réellement « chaud » ou « froid »).
3. Évaluation honnête (backtest walk-forward) : démontre empiriquement que
   les stratégies populaires (chauds / froids / en retard) ne battent pas,
   en moyenne, une sélection purement aléatoire.
4. Optimiseur de valeur espérée : génère des grilles qui NE changent PAS la
   probabilité de gagner, mais réduisent le risque de *partager* le jackpot
   en évitant les combinaisons populaires (biais des dates de naissance,
   suites, motifs réguliers). C'est la seule « optimisation » honnête au loto.

Dépendances : bibliothèque standard Python uniquement (aucun numpy/pandas),
pour être exécutable partout, y compris sur le serveur de production.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Règles du jeu EuroMillions (format depuis septembre 2016)
# ---------------------------------------------------------------------------
MAIN_MIN, MAIN_MAX, MAIN_COUNT = 1, 50, 5
STAR_MIN, STAR_MAX, STAR_COUNT = 1, 12, 2


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Draw:
    """Un tirage : 5 numéros triés + 2 étoiles triées, date optionnelle."""

    numbers: Tuple[int, ...]
    stars: Tuple[int, ...]
    date: Optional[str] = None


@dataclass
class Report:
    n_draws: int
    odds_jackpot: int
    n_modern_era: int = 0
    number_frequencies: Dict[int, int] = field(default_factory=dict)
    star_frequencies: Dict[int, int] = field(default_factory=dict)
    chi2_numbers: Tuple[float, int, float] = (0.0, 0, 1.0)
    chi2_stars: Tuple[float, int, float] = (0.0, 0, 1.0)
    sum_stats: Dict[str, float] = field(default_factory=dict)
    even_odd: Dict[str, float] = field(default_factory=dict)
    backtest: Dict[str, Dict[str, float]] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "n_draws": self.n_draws,
            "odds_jackpot": self.odds_jackpot,
            "n_modern_era": self.n_modern_era,
            "number_frequencies": self.number_frequencies,
            "star_frequencies": self.star_frequencies,
            "chi2_numbers": {
                "statistic": self.chi2_numbers[0],
                "dof": self.chi2_numbers[1],
                "p_value": self.chi2_numbers[2],
            },
            "chi2_stars": {
                "statistic": self.chi2_stars[0],
                "dof": self.chi2_stars[1],
                "p_value": self.chi2_stars[2],
            },
            "sum_stats": self.sum_stats,
            "even_odd": self.even_odd,
            "backtest": self.backtest,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
def load_draws(csv_path: str | Path) -> List[Draw]:
    """Charge et valide les tirages depuis le CSV (N1..N5, E1, E2, Date)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    draws: List[Draw] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            try:
                numbers = tuple(sorted(int(raw[f"N{i}"]) for i in range(1, 6)))
                stars = tuple(sorted(int(raw[f"E{i}"]) for i in range(1, 3)))
            except (KeyError, ValueError, TypeError):
                continue
            if not _valid(numbers, MAIN_MIN, MAIN_MAX, MAIN_COUNT):
                continue
            if not _valid(stars, STAR_MIN, STAR_MAX, STAR_COUNT):
                continue
            date = (raw.get("Date") or "").strip() or None
            draws.append(Draw(numbers=numbers, stars=stars, date=date))
    return draws


def _valid(values: Sequence[int], lo: int, hi: int, count: int) -> bool:
    return (
        len(values) == count
        and len(set(values)) == count
        and all(lo <= v <= hi for v in values)
    )


def modern_era_segment(draws: Sequence[Draw]) -> List[Draw]:
    """Retourne les tirages du format actuel (12 étoiles, depuis sept. 2016).

    Le nombre d'étoiles EuroMillions a évolué (9 → 11 → 12). Tester
    l'uniformité sur tout l'historique mélange des règles différentes et
    fausse le résultat. On isole le segment contigu utilisant la 12e étoile,
    quel que soit l'ordre (plus récent ou plus ancien en premier) du fichier.
    """
    idxs = [i for i, d in enumerate(draws) if STAR_MAX in d.stars]
    if not idxs:
        return list(draws)
    first, last = min(idxs), max(idxs)
    recent_first = first <= (len(draws) - 1 - last)
    return list(draws[: last + 1]) if recent_first else list(draws[first:])


# ---------------------------------------------------------------------------
# Probabilités de référence
# ---------------------------------------------------------------------------
def jackpot_odds() -> int:
    """1 chance sur C(50,5) * C(12,2) de gagner le jackpot."""
    return math.comb(MAIN_MAX, MAIN_COUNT) * math.comb(STAR_MAX, STAR_COUNT)


def expected_matches_random() -> Tuple[float, float]:
    """Nombre moyen de bons numéros / étoiles d'une grille aléatoire.

    Espérance = (tirés) * (choisis) / (univers). Indépendant de la stratégie
    tant que la sélection est faite sans information prédictive réelle.
    """
    main = MAIN_COUNT * MAIN_COUNT / MAIN_MAX
    star = STAR_COUNT * STAR_COUNT / STAR_MAX
    return main, star


# ---------------------------------------------------------------------------
# Statistiques descriptives
# ---------------------------------------------------------------------------
def frequencies(draws: Sequence[Draw]) -> Tuple[Counter, Counter]:
    nums: Counter = Counter()
    stars: Counter = Counter()
    for d in draws:
        nums.update(d.numbers)
        stars.update(d.stars)
    return nums, stars


def sum_statistics(draws: Sequence[Draw]) -> Dict[str, float]:
    sums = [sum(d.numbers) for d in draws]
    if not sums:
        return {}
    mean = sum(sums) / len(sums)
    var = sum((s - mean) ** 2 for s in sums) / len(sums)
    return {
        "min": float(min(sums)),
        "max": float(max(sums)),
        "mean": round(mean, 2),
        "std": round(math.sqrt(var), 2),
    }


def even_odd_ratio(draws: Sequence[Draw]) -> Dict[str, float]:
    even = sum(1 for d in draws for n in d.numbers if n % 2 == 0)
    total = len(draws) * MAIN_COUNT
    if total == 0:
        return {}
    return {
        "even_pct": round(100 * even / total, 2),
        "odd_pct": round(100 * (total - even) / total, 2),
    }


def current_gaps(draws: Sequence[Draw]) -> Dict[int, int]:
    """Nombre de tirages depuis la dernière sortie de chaque numéro."""
    gaps: Dict[int, int] = {n: len(draws) for n in range(MAIN_MIN, MAIN_MAX + 1)}
    for distance, draw in enumerate(reversed(draws)):
        for n in draw.numbers:
            if gaps[n] == len(draws):
                gaps[n] = distance
    return gaps


# ---------------------------------------------------------------------------
# Test du chi² d'uniformité (implémentation pure, sans scipy)
# ---------------------------------------------------------------------------
def chi_square_uniformity(
    counts: Counter, universe_size: int, n_draws: int, picks_per_draw: int
) -> Tuple[float, int, float]:
    """Test d'ajustement du chi² à une loi uniforme.

    H0 : chaque valeur a la même probabilité de sortir (tirage équitable).
    Retourne (statistique, degrés de liberté, p-value).
    Une p-value élevée (> 0.05) = compatible avec un tirage parfaitement
    équitable : les écarts de fréquence observés sont du bruit statistique.
    """
    expected = n_draws * picks_per_draw / universe_size
    if expected <= 0:
        return 0.0, 0, 1.0
    chi2 = 0.0
    for value in range(1, universe_size + 1):
        observed = counts.get(value, 0)
        chi2 += (observed - expected) ** 2 / expected
    dof = universe_size - 1
    p_value = _chi2_sf(chi2, dof)
    return round(chi2, 4), dof, round(p_value, 6)


def _chi2_sf(x: float, k: int) -> float:
    """Fonction de survie du chi² : P(X > x) = Q(k/2, x/2)."""
    if x <= 0:
        return 1.0
    return _gammq(k / 2.0, x / 2.0)


def _gammq(a: float, x: float) -> float:
    """Fonction gamma incomplète régularisée supérieure Q(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("arguments invalides pour gammq")
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def _gser(a: float, x: float) -> float:
    """Série pour P(a, x), valable pour x < a + 1."""
    gln = math.lgamma(a)
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    """Fraction continue pour Q(a, x), valable pour x >= a + 1."""
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


# ---------------------------------------------------------------------------
# Évaluation honnête : backtest walk-forward
# ---------------------------------------------------------------------------
def _top_n(counter: Counter, universe: range, n: int, reverse: bool) -> List[int]:
    ranked = sorted(universe, key=lambda v: (counter.get(v, 0), v), reverse=reverse)
    return ranked[:n]


def walk_forward_backtest(
    draws: Sequence[Draw],
    test_size: int = 400,
    rng_seed: int = 12345,
) -> Dict[str, Dict[str, float]]:
    """Mesure le nombre moyen de bons numéros par tirage pour 4 stratégies.

    Protocole walk-forward sans fuite de données : pour prédire le tirage t,
    on n'utilise que les tirages 0..t-1. On compare :
      - random  : 5 numéros au hasard (référence)
      - hot     : les 5 numéros les plus fréquents jusqu'ici
      - cold    : les 5 numéros les moins fréquents jusqu'ici
      - due     : les 5 numéros avec le plus grand écart (« en retard »)

    Résultat attendu si le jeu est équitable : toutes les stratégies
    obtiennent en moyenne ~0.5 bon numéro par tirage, comme le hasard.
    """
    n = len(draws)
    if n < test_size + 50:
        test_size = max(1, n // 3)
    start = n - test_size
    rng = random.Random(rng_seed)
    universe = range(MAIN_MIN, MAIN_MAX + 1)

    totals = {k: 0 for k in ("random", "hot", "cold", "due")}
    running: Counter = Counter()
    last_seen: Dict[int, int] = {v: -1 for v in universe}
    for i in range(start):
        running.update(draws[i].numbers)
        for v in draws[i].numbers:
            last_seen[v] = i

    for t in range(start, n):
        actual = set(draws[t].numbers)

        pick_random = rng.sample(list(universe), MAIN_COUNT)
        pick_hot = _top_n(running, universe, MAIN_COUNT, reverse=True)
        pick_cold = _top_n(running, universe, MAIN_COUNT, reverse=False)
        due_order = sorted(universe, key=lambda v: (last_seen[v], v))
        pick_due = due_order[:MAIN_COUNT]

        totals["random"] += len(actual & set(pick_random))
        totals["hot"] += len(actual & set(pick_hot))
        totals["cold"] += len(actual & set(pick_cold))
        totals["due"] += len(actual & set(pick_due))

        running.update(draws[t].numbers)
        for v in draws[t].numbers:
            last_seen[v] = t

    tested = n - start
    expected_main, _ = expected_matches_random()
    return {
        name: {
            "avg_matches": round(total / tested, 4),
            "expected_random": round(expected_main, 4),
            "tested_draws": tested,
        }
        for name, total in totals.items()
    }


# ---------------------------------------------------------------------------
# Optimiseur de valeur espérée (ne change PAS la probabilité de gagner)
# ---------------------------------------------------------------------------
def popularity_score(numbers: Sequence[int]) -> float:
    """Score heuristique de « popularité » d'une grille (0 = rare, 1 = très jouée).

    Les joueurs choisissent massivement : des dates (<= 31), des suites, des
    motifs réguliers, des grilles de somme « moyenne ». Jouer une grille rare
    ne change pas la probabilité de gagner, mais réduit le nombre de gagnants
    avec qui partager le jackpot → espérance de gain plus élevée.
    """
    nums = sorted(numbers)
    score = 0.0

    # Biais des dates de naissance : tous les numéros <= 31.
    if all(n <= 31 for n in nums):
        score += 0.45
    else:
        frac = sum(1 for n in nums if n <= 31) / len(nums)
        score += 0.20 * frac

    # Suites arithmétiques / consécutifs (1-2-3-4-5, 10-20-30-40-50...).
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    if len(set(diffs)) == 1:
        score += 0.25
    consecutive = sum(1 for d in diffs if d == 1)
    score += 0.05 * consecutive

    # Somme proche de la moyenne « populaire » (~128) choisie par beaucoup.
    total = sum(nums)
    score += 0.15 * max(0.0, 1.0 - abs(total - 128) / 60.0)

    # Numéros tous dans la même dizaine (motif visuel).
    if max(nums) - min(nums) <= 10:
        score += 0.10

    return round(min(score, 1.0), 4)


def generate_ev_grids(
    n_grids: int = 5,
    rng_seed: Optional[int] = None,
    max_popularity: float = 0.25,
    attempts: int = 20000,
) -> List[Dict[str, object]]:
    """Génère des grilles aléatoires filtrées pour être peu populaires.

    IMPORTANT : chaque grille a exactement la même probabilité de gagner que
    n'importe quelle autre (1 / {:,}). Le seul effet du filtre est de réduire
    la probabilité de partage du jackpot.
    """.format(jackpot_odds())
    rng = random.Random(rng_seed)
    main_universe = list(range(MAIN_MIN, MAIN_MAX + 1))
    star_universe = list(range(STAR_MIN, STAR_MAX + 1))

    grids: List[Dict[str, object]] = []
    seen = set()
    tries = 0
    while len(grids) < n_grids and tries < attempts:
        tries += 1
        numbers = tuple(sorted(rng.sample(main_universe, MAIN_COUNT)))
        if numbers in seen:
            continue
        score = popularity_score(numbers)
        if score > max_popularity:
            continue
        seen.add(numbers)
        stars = tuple(sorted(rng.sample(star_universe, STAR_COUNT)))
        grids.append(
            {
                "numbers": list(numbers),
                "stars": list(stars),
                "popularity": score,
                "sum": sum(numbers),
            }
        )

    # Repli : si le filtre est trop strict, compléter avec les moins populaires.
    if len(grids) < n_grids:
        pool = []
        for _ in range(attempts):
            numbers = tuple(sorted(rng.sample(main_universe, MAIN_COUNT)))
            if numbers in seen:
                continue
            seen.add(numbers)
            pool.append((popularity_score(numbers), numbers))
        pool.sort(key=lambda x: x[0])
        for score, numbers in pool:
            if len(grids) >= n_grids:
                break
            stars = tuple(sorted(rng.sample(star_universe, STAR_COUNT)))
            grids.append(
                {
                    "numbers": list(numbers),
                    "stars": list(stars),
                    "popularity": score,
                    "sum": sum(numbers),
                }
            )
    return grids


# ---------------------------------------------------------------------------
# Rapport complet
# ---------------------------------------------------------------------------
def build_report(draws: Sequence[Draw], test_size: int = 400) -> Report:
    nums, stars = frequencies(draws)
    # Le test d'équité ne porte que sur l'ère homogène (12 étoiles), sinon
    # le changement de règles (9 → 11 → 12 étoiles) fausse l'uniformité.
    modern = modern_era_segment(draws)
    nums_m, stars_m = frequencies(modern)
    note = (
        f"Test d'équité calculé sur les {len(modern)} tirages du format actuel "
        f"(5 numéros/50, 2 étoiles/12). L'historique antérieur utilisait 9 puis "
        f"11 étoiles : l'inclure fausserait le test d'uniformité des étoiles."
    )
    return Report(
        n_draws=len(draws),
        odds_jackpot=jackpot_odds(),
        n_modern_era=len(modern),
        number_frequencies=dict(sorted(nums.items())),
        star_frequencies=dict(sorted(stars.items())),
        chi2_numbers=chi_square_uniformity(nums_m, MAIN_MAX, len(modern), MAIN_COUNT),
        chi2_stars=chi_square_uniformity(stars_m, STAR_MAX, len(modern), STAR_COUNT),
        sum_stats=sum_statistics(draws),
        even_odd=even_odd_ratio(draws),
        backtest=walk_forward_backtest(draws, test_size=test_size),
        note=note,
    )


def format_report(report: Report) -> str:
    lines: List[str] = []
    add = lines.append
    add("=" * 70)
    add("ANALYSE STATISTIQUE HONNÊTE — EUROMILLIONS")
    add("=" * 70)
    add(f"Tirages analysés          : {report.n_draws}")
    add(f"Probabilité du jackpot     : 1 sur {report.odds_jackpot:,}".replace(",", " "))
    exp_main, exp_star = expected_matches_random()
    add(f"Bons numéros/tirage (hasard): {exp_main:.3f}  |  étoiles : {exp_star:.3f}")
    add("")

    add("TEST DU CHI² D'UNIFORMITÉ (le tirage est-il équitable ?)")
    add("-" * 70)
    add(f"  Base : {report.n_modern_era} tirages (format actuel 12 étoiles)")
    c2n = report.chi2_numbers
    c2s = report.chi2_stars
    add(f"  Numéros : chi²={c2n[0]:.2f}  ddl={c2n[1]}  p-value={c2n[2]:.4f}")
    add(f"  Étoiles : chi²={c2s[0]:.2f}  ddl={c2s[1]}  p-value={c2s[2]:.4f}")
    verdict_n = "ÉQUITABLE (compatible hasard)" if c2n[2] > 0.05 else "écart notable"
    verdict_s = "ÉQUITABLE (compatible hasard)" if c2s[2] > 0.05 else "écart notable"
    add(f"  Verdict numéros : {verdict_n}")
    add(f"  Verdict étoiles : {verdict_s}")
    add("")

    add("ÉVALUATION HONNÊTE — aucune stratégie ne bat le hasard")
    add("-" * 70)
    add(f"  {'stratégie':<10} {'moy. bons n°':>14} {'attendu hasard':>16}")
    for name, stats in report.backtest.items():
        add(
            f"  {name:<10} {stats['avg_matches']:>14.4f} "
            f"{stats['expected_random']:>16.4f}"
        )
    add("")
    add("  → Les stratégies « chauds/froids/en retard » convergent vers la même")
    add("    valeur que le tirage aléatoire. Elles n'ont aucun pouvoir prédictif.")
    add("")

    add("SOMMES & PARITÉ")
    add("-" * 70)
    if report.sum_stats:
        s = report.sum_stats
        add(f"  Somme des 5 numéros : min={s['min']:.0f} max={s['max']:.0f} "
            f"moy={s['mean']:.1f} écart-type={s['std']:.1f}")
    if report.even_odd:
        add(f"  Pairs : {report.even_odd['even_pct']:.1f}%  "
            f"Impairs : {report.even_odd['odd_pct']:.1f}%")
    add("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interface ligne de commande
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyse statistique honnête de l'EuroMillions (aucune "
        "prédiction : le tirage est aléatoire et imprévisible).",
    )
    parser.add_argument(
        "--csv", default="tirage_euromillions_complet.csv", help="Fichier CSV des tirages"
    )
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    parser.add_argument(
        "--test-size", type=int, default=400, help="Taille de la fenêtre de backtest"
    )
    parser.add_argument(
        "--generate", type=int, default=0, metavar="N",
        help="Générer N grilles optimisées en valeur espérée (ne change pas "
        "la probabilité de gagner)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Graine aléatoire")
    args = parser.parse_args(argv)

    draws = load_draws(args.csv)
    report = build_report(draws, test_size=args.test_size)

    payload: dict = report.to_dict()
    if args.generate > 0:
        payload["ev_grids"] = generate_ev_grids(args.generate, rng_seed=args.seed)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
        if args.generate > 0:
            print("\nGRILLES OPTIMISÉES EN VALEUR ESPÉRÉE")
            print("-" * 70)
            print("  (même probabilité de gagner, moins de risque de partage)")
            for i, g in enumerate(payload["ev_grids"], 1):
                nums = " ".join(f"{n:2d}" for n in g["numbers"])
                st = " ".join(f"{s:2d}" for s in g["stars"])
                print(f"  #{i}: {nums}  ★ {st}   (popularité={g['popularity']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
