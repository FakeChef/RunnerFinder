"""V1.3 integration: the S3 OpenMP fail-safe (ocr/diagnostic._run_photos + scanner/pipeline3.s3_preconditions).

S3 may run only with the configuration validated in V1.3 Phase B (the shared Intel OpenMP runtime loaded and
usable, KMP_BLOCKTIME=0). Otherwise the V1.2 prefetch pipeline runs and the reason is reported."""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

# Imported lazily inside the tests: unittest discovery imports every test module into ONE process, and the GUI
# tests (test_search_integration ...zz) check that no OCR / detection module is loaded in that process.
diagnostic = pipeline3 = None


def _engine():
    global diagnostic, pipeline3
    if diagnostic is None:
        from ocr import diagnostic as d
        from scanner import pipeline3 as p
        diagnostic, pipeline3 = d, p


class FakeDiag:
    pipeline_used = ""
    pipeline_fallback = ""


class RunPhotosFailSafeTest(unittest.TestCase):
    def setUp(self):
        _engine()

    def run_photos(self, *, torch_loaded, hook_ok, kmp):
        used = {}

        def fake_loop(diag, cfg, pipeline, selected, run_dir, debug, progress, store, on_decoded, photos):
            used["photos"] = type(photos).__name__
        info, diag = {}, FakeDiag()
        finder = types.SimpleNamespace(find=lambda image: None)
        modules = dict(sys.modules)
        if torch_loaded:
            modules.setdefault("torch", types.ModuleType("torch"))
        else:
            modules.pop("torch", None)
        hook = (lambda: None) if hook_ok else None
        env = dict(os.environ)
        env.pop("KMP_BLOCKTIME", None)
        if kmp is not None:
            env["KMP_BLOCKTIME"] = kmp
        with mock.patch.dict(sys.modules, modules, clear=True), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(diagnostic, "_photo_loop", fake_loop), \
                mock.patch.object(pipeline3, "openmp_single_thread_hook",
                                  lambda: (hook, "ok" if hook_ok else "libiomp5md.dll is not loaded in this process")), \
                mock.patch.object(pipeline3, "PhotoPrefetcher") as pf:
            pf.return_value.__enter__ = lambda s: types.SimpleNamespace(kind="prefetch")
            pf.return_value.__exit__ = lambda s, *a: None
            diagnostic._run_photos(diag, None, types.SimpleNamespace(finder=finder), [], None, False, None, None,
                                   None, prefetch=2, three_stage=True, pipeline_info=info)
        return info, used

    def test_validated_configuration_runs_s3(self):
        info, used = self.run_photos(torch_loaded=True, hook_ok=True, kmp="0")
        self.assertEqual(info, {"pipeline_used": "s3", "pipeline_fallback": ""})
        self.assertEqual(used["photos"], "DetectionPipeline")

    def test_missing_openmp_runtime_falls_back_to_prefetch(self):
        info, used = self.run_photos(torch_loaded=True, hook_ok=False, kmp="0")
        self.assertEqual(info["pipeline_used"], "prefetch")
        self.assertIn("libiomp5md.dll", info["pipeline_fallback"])
        self.assertNotEqual(used["photos"], "DetectionPipeline")

    def test_unverified_kmp_blocktime_falls_back_to_prefetch(self):
        for kmp in (None, "200", "1"):
            info, used = self.run_photos(torch_loaded=True, hook_ok=True, kmp=kmp)
            self.assertEqual(info["pipeline_used"], "prefetch", kmp)
            self.assertIn("KMP_BLOCKTIME", info["pipeline_fallback"], kmp)
            self.assertNotEqual(used["photos"], "DetectionPipeline", kmp)

    def test_stand_in_detectors_without_torch_keep_s3(self):
        info, used = self.run_photos(torch_loaded=False, hook_ok=False, kmp=None)
        self.assertEqual(info["pipeline_used"], "s3")          # tests with stand-in detectors (no OpenMP involved)

    def test_prefetch_zero_is_sequential(self):
        info = {}
        with mock.patch.object(diagnostic, "_photo_loop", lambda *a: None):
            diagnostic._run_photos(FakeDiag(), None, None, [], None, False, None, None, None, prefetch=0,
                                   three_stage=True, pipeline_info=info)
        self.assertEqual(info["pipeline_used"], "sequential")


class PreconditionsTest(unittest.TestCase):
    def setUp(self):
        _engine()

    def test_no_torch_no_hook(self):
        with mock.patch.dict(sys.modules, {k: v for k, v in sys.modules.items() if k != "torch"}, clear=True):
            hook, reason = pipeline3.openmp_single_thread_hook()
        self.assertIsNone(hook)
        self.assertEqual(reason, "torch not loaded")

    def test_kmp_must_be_validated_value(self):
        with mock.patch.object(pipeline3, "openmp_single_thread_hook", lambda: ((lambda: None), "ok")):
            with mock.patch.dict(os.environ, {"KMP_BLOCKTIME": "0"}):
                self.assertEqual(pipeline3.s3_preconditions(), (True, "ok"))
            with mock.patch.dict(os.environ, {"KMP_BLOCKTIME": "200"}):
                ok, reason = pipeline3.s3_preconditions()
                self.assertFalse(ok)
                self.assertIn("'200'", reason)


if __name__ == "__main__":
    unittest.main()
