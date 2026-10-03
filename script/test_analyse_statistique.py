#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests unitaires du module d'analyse statistique honnête.

Exécution :  python3 -m unittest script.test_analyse_statistique
         ou  python3 script/test_analyse_statistique.py
"""

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse_statistique as a  # noqa: E402


def _random_draws(n, seed=0):
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        nums = tuple(sorted(rng.sample(range(a.MAIN_MIN, a.MAIN_MAX + 1), a.MAIN_COUNT)))
        stars = tuple(sorted(rng.sample(range(a.STAR_MIN, a.STAR_MAX + 1), a.STAR_COUNT)))
        draws.append(a.Draw(numbers=nums, stars=stars))
    return draws


class TestConstantsAndOdds(unittest.TestCase):
    def test_jackpot_odds(self):
        # C(50,5) * C(12,2) = 2 118 760 * 66 = 139 838 160
        self.assertEqual(a.jackpot_odds(), 139_838_160)

    def test_expected_matches_random(self):
        main, star = a.expected_matches_random()
        self.assertAlmostEqual(main, 0.5, places=6)
        self.assertAlmostEqual(star, 2 / 6, places=6)


class TestChiSquare(unittest.TestCase):
    def test_sf_known_values(self):
        # P(chi2 > dof) pour un chi2 égal aux degrés de liberté est ~0.3-0.5.
        self.assertTrue(0.2 < a._chi2_sf(49, 49) < 0.6)
        # Un chi2 très grand donne une p-value quasi nulle.
        self.assertLess(a._chi2_sf(200, 49), 1e-6)
        # x=0 -> p-value = 1.
        self.assertEqual(a._chi2_sf(0, 10), 1.0)

    def test_uniform_data_passes(self):
        draws = _random_draws(1500, seed=1)
        nums, _ = a.frequencies(draws)
        _, _, p = a.chi_square_uniformity(nums, a.MAIN_MAX, len(draws), a.MAIN_COUNT)
        self.assertGreater(p, 0.01, "des tirages aléatoires doivent passer le test")

    def test_biased_data_fails(self):
        # Univers biaisé : seuls 10 numéros sortent jamais.
        from collections import Counter

        biased = Counter({i: 1000 for i in range(1, 11)})
        _, _, p = a.chi_square_uniformity(biased, a.MAIN_MAX, 2000, a.MAIN_COUNT)
        self.assertLess(p, 1e-6, "un univers biaisé doit échouer au test")


class TestPopularity(unittest.TestCase):
    def test_birthday_grid_is_popular(self):
        self.assertGreater(a.popularity_score([3, 12, 17, 24, 31]), 0.4)

    def test_arithmetic_sequence_is_popular(self):
        self.assertGreater(a.popularity_score([10, 20, 30, 40, 50]), 0.2)

    def test_consecutive_run_is_popular(self):
        self.assertGreater(a.popularity_score([1, 2, 3, 4, 5]), 0.4)

    def test_spread_grid_is_rare(self):
        self.assertLess(a.popularity_score([3, 19, 34, 41, 48]), 0.3)


class TestEvGrids(unittest.TestCase):
    def test_grids_valid_and_unpopular(self):
        grids = a.generate_ev_grids(5, rng_seed=42, max_popularity=0.25)
        self.assertEqual(len(grids), 5)
        seen = set()
        for g in grids:
            nums, stars = g["numbers"], g["stars"]
            self.assertEqual(len(nums), a.MAIN_COUNT)
            self.assertEqual(len(set(nums)), a.MAIN_COUNT)
            self.assertTrue(all(a.MAIN_MIN <= n <= a.MAIN_MAX for n in nums))
            self.assertEqual(len(stars), a.STAR_COUNT)
            self.assertTrue(all(a.STAR_MIN <= s <= a.STAR_MAX for s in stars))
            self.assertLessEqual(g["popularity"], 0.25)
            seen.add(tuple(nums))
        self.assertEqual(len(seen), 5, "les grilles doivent être distinctes")


class TestBacktestHonesty(unittest.TestCase):
    def test_no_strategy_beats_random(self):
        """Sur des tirages aléatoires, aucune stratégie ne dépasse ~0.5."""
        draws = _random_draws(1500, seed=7)
        result = a.walk_forward_backtest(draws, test_size=500, rng_seed=99)
        for name, stats in result.items():
            self.assertLess(
                stats["avg_matches"], 0.75,
                f"{name} ne devrait pas battre significativement le hasard",
            )
            self.assertGreaterEqual(stats["avg_matches"], 0.25)


class TestLoader(unittest.TestCase):
    def test_load_and_skip_malformed(self):
        content = (
            "N1,N2,N3,N4,N5,E1,E2,Date\n"
            "2,4,15,21,48,6,12,2025-11-18\n"   # valide
            "9,26,27,45,48,8,9,\n"             # valide
            "1,1,2,3,4,5,6,\n"                 # doublon -> ignoré
            "1,2,3,4,99,5,6,\n"               # hors borne -> ignoré
            "x,y,z,,,,,\n"                     # non numérique -> ignoré
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(content)
            path = fh.name
        try:
            draws = a.load_draws(path)
            self.assertEqual(len(draws), 2)
            self.assertEqual(draws[0].numbers, (2, 4, 15, 21, 48))
            self.assertEqual(draws[0].stars, (6, 12))
        finally:
            os.unlink(path)


class TestModernEra(unittest.TestCase):
    def test_detects_recent_first_segment(self):
        # 10 tirages 12-étoiles récents, puis 10 anciens plafonnés à 9 étoiles.
        recent = [a.Draw((1, 2, 3, 4, 5), (11, 12)) for _ in range(10)]
        old = [a.Draw((1, 2, 3, 4, 5), (8, 9)) for _ in range(10)]
        seg = a.modern_era_segment(recent + old)
        self.assertEqual(len(seg), 10)
        self.assertTrue(all(12 in d.stars for d in seg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
