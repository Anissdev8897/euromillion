#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests des modules de sécurité (api_security, safe_model_loader).

Bibliothèque standard uniquement — exécutable sans Flask/joblib installés.
    python3 script/test_security.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import api_security as sec  # noqa: E402
import safe_model_loader as sml  # noqa: E402


class _FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestApiKey(unittest.TestCase):
    def setUp(self):
        for k in ("API_KEY", "EUROMILLIONS_API_KEY"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("API_KEY", "EUROMILLIONS_API_KEY"):
            os.environ.pop(k, None)

    def test_constant_time_compare(self):
        self.assertTrue(sec.constant_time_compare("abc", "abc"))
        self.assertFalse(sec.constant_time_compare("abc", "abd"))
        self.assertFalse(sec.constant_time_compare(None, "abc"))
        self.assertFalse(sec.constant_time_compare("abc", None))

    def test_auth_disabled_by_default(self):
        ok, reason = sec.check_api_key(None)
        self.assertTrue(ok)
        self.assertEqual(reason, "auth-disabled")
        self.assertFalse(sec.auth_enabled())

    def test_auth_enforced_when_configured(self):
        os.environ["API_KEY"] = "s3cr3t"
        self.assertTrue(sec.auth_enabled())
        self.assertEqual(sec.check_api_key("s3cr3t"), (True, "ok"))
        self.assertEqual(sec.check_api_key("wrong")[0], False)
        self.assertEqual(sec.check_api_key(None)[0], False)


class TestCors(unittest.TestCase):
    def setUp(self):
        os.environ.pop("CORS_ORIGINS", None)

    def tearDown(self):
        os.environ.pop("CORS_ORIGINS", None)

    def test_default_origins(self):
        origins = sec.parse_cors_origins("")
        self.assertIn("https://kenopredictionia.fr", origins)

    def test_csv_parsing(self):
        origins = sec.parse_cors_origins("https://a.fr, https://b.fr")
        self.assertEqual(origins, ["https://a.fr", "https://b.fr"])

    def test_wildcard_and_exact(self):
        allowed = ["https://kenopredictionia.fr", "http://localhost:*"]
        self.assertTrue(sec.is_origin_allowed("https://kenopredictionia.fr", allowed))
        self.assertTrue(sec.is_origin_allowed("http://localhost:3000", allowed))
        self.assertTrue(sec.is_origin_allowed("http://localhost:8080", allowed))
        self.assertFalse(sec.is_origin_allowed("https://evil.com", allowed))
        self.assertFalse(sec.is_origin_allowed(None, allowed))

    def test_star_matches_all(self):
        self.assertTrue(sec.is_origin_allowed("https://anything.com", ["*"]))


class TestRateLimiter(unittest.TestCase):
    def test_blocks_after_limit(self):
        clock = _FakeClock()
        rl = sec.RateLimiter(max_requests=3, window_seconds=60, clock=clock)
        self.assertTrue(rl.allow("ip1"))
        self.assertTrue(rl.allow("ip1"))
        self.assertTrue(rl.allow("ip1"))
        self.assertFalse(rl.allow("ip1"))  # 4e requête bloquée
        self.assertGreaterEqual(rl.retry_after("ip1"), 1)

    def test_window_slides(self):
        clock = _FakeClock()
        rl = sec.RateLimiter(max_requests=2, window_seconds=60, clock=clock)
        self.assertTrue(rl.allow("ip"))
        self.assertTrue(rl.allow("ip"))
        self.assertFalse(rl.allow("ip"))
        clock.advance(61)  # la fenêtre a glissé
        self.assertTrue(rl.allow("ip"))

    def test_keys_isolated(self):
        clock = _FakeClock()
        rl = sec.RateLimiter(max_requests=1, window_seconds=60, clock=clock)
        self.assertTrue(rl.allow("a"))
        self.assertFalse(rl.allow("a"))
        self.assertTrue(rl.allow("b"))  # une autre IP n'est pas affectée


class TestSanitizedError(unittest.TestCase):
    def test_shape_and_no_leak(self):
        err = sec.sanitized_error()
        self.assertEqual(err["status"], "error")
        self.assertIn("error_id", err)
        self.assertEqual(len(err["error_id"]), 12)
        # Message générique, pas de trace ni de chemin.
        self.assertNotIn("/", err["message"])


class TestSafeModelLoader(unittest.TestCase):
    def setUp(self):
        os.environ.pop("EUROMILLIONS_STRICT_MODELS", None)
        self.tmp = tempfile.mkdtemp()
        self.model = Path(self.tmp) / "model.pkl"
        self.model.write_bytes(b"not-a-real-model-payload")
        self.digest = sml.sha256_file(self.model)

    def tearDown(self):
        os.environ.pop("EUROMILLIONS_STRICT_MODELS", None)

    def _manifest(self, mapping):
        path = Path(self.tmp) / "manifest.json"
        path.write_text(json.dumps({"files": mapping}), encoding="utf-8")
        return path

    def test_sha256_matches_hashlib(self):
        import hashlib
        expected = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.assertEqual(self.digest, expected)

    def test_verify_ok_and_mismatch(self):
        man = sml.load_manifest(self._manifest({"model.pkl": self.digest}))
        self.assertEqual(sml.verify_file(self.model, man), (True, "ok"))
        bad = sml.load_manifest(self._manifest({"model.pkl": "0" * 64}))
        self.assertEqual(sml.verify_file(self.model, bad)[1], "hash-different")

    def test_verify_missing_cases(self):
        self.assertEqual(sml.verify_file(self.model, None)[1], "manifeste-absent")
        man = sml.load_manifest(self._manifest({"autre.pkl": self.digest}))
        self.assertEqual(sml.verify_file(self.model, man)[1], "hash-non-reference")
        self.assertEqual(
            sml.verify_file(Path(self.tmp) / "absent.pkl", man)[1], "fichier-absent"
        )

    def test_safe_load_ok(self):
        man = self._manifest({"model.pkl": self.digest})
        calls = []
        result = sml.safe_load(self.model, lambda p: calls.append(p) or "LOADED",
                               manifest_path=man)
        self.assertEqual(result, "LOADED")
        self.assertEqual(len(calls), 1)

    def test_safe_load_strict_refuses_unverified(self):
        man = self._manifest({"autre.pkl": self.digest})
        with self.assertRaises(sml.ModelIntegrityError):
            sml.safe_load(self.model, lambda p: "LOADED", manifest_path=man, strict=True)

    def test_safe_load_non_strict_loads_with_warning(self):
        man = self._manifest({"autre.pkl": self.digest})
        result = sml.safe_load(self.model, lambda p: "LOADED", manifest_path=man, strict=False)
        self.assertEqual(result, "LOADED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
