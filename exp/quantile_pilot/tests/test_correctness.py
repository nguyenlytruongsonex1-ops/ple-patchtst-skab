"""Correctness locks for quantile PLE, RevIN, SKAB windows, and Bike windows.

Synthetic checks always run. Checks that read official files run when
``download_data.py`` has populated ``exp/data``. Nothing here trains the pilot.
"""

from __future__ import annotations

import inspect
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import run_detect  # noqa: E402
from quantile_pilot.config import (  # noqa: E402
    BIKE_HORIZON,
    BIKE_LOOKBACK,
    EQUAL_BIN_MODEL,
    N_BINS,
    PILOT_MODELS,
)
from quantile_pilot.data_bike import (  # noqa: E402
    BIKE_FEATURES,
    build_bike_dataset,
    seasonal_naive,
)
from quantile_pilot.data_skab import (  # noqa: E402
    FEATURES,
    groups_are_intact,
    load_skab_fold,
    load_splits,
    windows_from_frame,
)
from quantile_pilot.jobs import assert_pilot_size, plan_jobs  # noqa: E402
from quantile_pilot.models import (  # noqa: E402
    build_model,
    copy_shared_weights,
    denorm_channel,
    predict,
)
from quantile_pilot.ple import (  # noqa: E402
    QuantilePLE,
    fit_revin_quantile_edges,
    instance_norm,
    ple_encode,
)
from quantile_pilot.train import iter_batches, train_one  # noqa: E402
from layers.RevIN import RevIN  # noqa: E402

EXP = _EXP


def _bike_frame(n=80, gap_at=None, wind_zero_at=None, cnt_fn=None):
    start = pd.Timestamp("2011-01-01")
    stamps = pd.date_range(start, periods=n, freq="h")
    rows = []
    for i, ts in enumerate(stamps):
        if gap_at is not None and i in gap_at:
            continue
        cnt = float(i + 1 if cnt_fn is None else cnt_fn(i))
        rows.append(
            {
                "instant": i + 1,
                "dteday": ts.date().isoformat(),
                "season": 1 + (ts.month % 4),
                "yr": int(ts.year == 2012),
                "mnth": ts.month,
                "hr": ts.hour,
                "holiday": 0,
                "weekday": int(ts.day_of_week),
                "workingday": int(ts.day_of_week < 5),
                "weathersit": 1,
                "temp": 0.2 + (i % 5) * 0.01,
                "atemp": 0.25,
                "hum": 0.0 if wind_zero_at is not None and i in wind_zero_at else 0.4,
                "windspeed": 0.0 if wind_zero_at is not None and i in wind_zero_at else 0.1,
                "casual": cnt / 4,
                "registered": cnt * 3 / 4,
                "cnt": cnt,
            }
        )
    return pd.DataFrame(rows)


class PLETests(unittest.TestCase):
    def test_expected_values_and_gradient(self):
        edges = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
        ple = QuantilePLE(edges)
        with torch.no_grad():
            ple.W.fill_(4.0)
        x = torch.tensor([[[1.0]]], requires_grad=True)  # mid-bin, fraction 0.5
        out = ple(x)
        self.assertTrue(torch.allclose(out, torch.tensor([[[1.0 + 4.0 * 0.5]]])))
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.tensor([[[1.0 + 4.0 * 0.5]]])))
        self.assertTrue(torch.allclose(ple.W.grad, torch.tensor([[0.5]])))

        below = torch.tensor([[[-1.0]]])
        above = torch.tensor([[[5.0]]])
        self.assertTrue(torch.allclose(ple_encode(below, edges), torch.zeros(1, 1, 1)))
        self.assertTrue(torch.allclose(ple_encode(above, edges), torch.ones(1, 1, 1)))

    def test_duplicate_and_constant_are_finite(self):
        edges = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
        ple = QuantilePLE(edges)
        z = torch.tensor([[[-0.2, 0.4], [0.0, 2.0]]], requires_grad=True)
        out = ple(z)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.isfinite(ple.W.grad).all())
        # Tied edges are a step, still inside [0, 1].
        enc = ple.encode(z)
        self.assertTrue(torch.all((enc >= 0) & (enc <= 1)))

    def test_channel_weights_do_not_mix(self):
        edges = torch.linspace(0, 1, 4).repeat(2, 1)
        ple = QuantilePLE(edges)
        with torch.no_grad():
            ple.W.zero_()
            ple.W[1].fill_(1.0)
        z = torch.zeros(2, 3, 2, requires_grad=True)
        z.data[..., 0] = 0.2
        z.data[..., 1] = 0.2
        out = ple(z)
        self.assertTrue(torch.allclose(out[..., 0], z[..., 0]))
        self.assertFalse(torch.allclose(out[..., 1], z[..., 1]))

    def test_edges_match_revin_quantiles_not_raw(self):
        torch.manual_seed(0)
        raw = torch.arange(200, dtype=torch.float32).reshape(20, 10, 1) * 3 + 8
        edges = fit_revin_quantile_edges(raw, n_bins=4, chunk=3)
        edges_full = fit_revin_quantile_edges(raw, n_bins=4, chunk=100)
        self.assertTrue(torch.equal(edges, edges_full))
        z, _, _ = instance_norm(raw)
        probs = torch.linspace(0, 1, 5)
        manual = torch.quantile(z.reshape(-1), probs)
        self.assertTrue(torch.allclose(edges[0], manual))
        raw_q = torch.quantile(raw.reshape(-1), probs)
        self.assertFalse(torch.allclose(edges[0], raw_q))
        self.assertLess(float(edges.abs().mean()), 5.0)
        self.assertGreater(float(raw_q.abs().mean()), 20.0)

    def test_constant_sensor_and_ignored_labels(self):
        torch.manual_seed(1)
        x = torch.randn(12, 8, 2)
        x[:, :, 0] = 5.0
        edges = fit_revin_quantile_edges(x, n_bins=N_BINS)
        self.assertEqual(tuple(edges.shape), (2, N_BINS + 1))
        self.assertTrue(torch.allclose(edges[0], torch.zeros(N_BINS + 1), atol=1e-6))
        self.assertGreater(float(edges[1].max() - edges[1].min()), 0.1)
        # The fitter has no label argument. A separate val tensor cannot move edges.
        params = inspect.signature(fit_revin_quantile_edges).parameters
        self.assertNotIn("y", params)
        self.assertNotIn("val", params)
        val = x + 100
        again = fit_revin_quantile_edges(x, n_bins=N_BINS)
        self.assertTrue(torch.equal(edges, again))
        # A pure location shift is removed by RevIN, so the quantile edges do not move.
        self.assertTrue(
            torch.allclose(edges, fit_revin_quantile_edges(torch.cat([x, val], 0)), atol=1e-5)
        )
        shaped = x.clone()
        shaped[:, :, 1] = torch.linspace(-2, 2, x.shape[1])
        self.assertFalse(
            torch.allclose(edges[1], fit_revin_quantile_edges(torch.cat([x, shaped], 0))[1])
        )
        from rtdl_num_embeddings import compute_bins

        flat = x.reshape(-1, 2)
        with self.assertRaises(ValueError):
            compute_bins(flat, n_bins=4)

    def test_matches_rtdl_inside_and_clips_outside(self):
        from rtdl_num_embeddings import _PiecewiseLinearEncodingImpl

        edges = torch.tensor([[0.0, 1.0, 3.0], [0.0, 2.0, 4.0]], dtype=torch.float32)
        impl = _PiecewiseLinearEncodingImpl([edges[0], edges[1]])
        interior = torch.tensor([[0.5, 1.0], [2.0, 3.0]])
        ours = ple_encode(interior, edges)
        ref = impl(interior)
        self.assertTrue(torch.allclose(ours, ref, atol=1e-6))
        outside = torch.tensor([[-2.0, -3.0], [9.0, 8.0]])
        clipped = ple_encode(outside, edges)
        extrapolated = impl(outside)
        self.assertTrue(torch.all((clipped >= 0) & (clipped <= 1)))
        self.assertTrue(bool((extrapolated < 0).any() or (extrapolated > 1).any()))

    def test_serialization_roundtrip(self):
        torch.manual_seed(2)
        edges = fit_revin_quantile_edges(torch.randn(6, 5, 3), n_bins=4)
        ple = QuantilePLE(edges)
        with torch.no_grad():
            ple.W.normal_()
        blob = io.BytesIO()
        torch.save(ple.state_dict(), blob)
        blob.seek(0)
        restored = QuantilePLE(torch.zeros_like(edges))
        restored.load_state_dict(torch.load(blob, weights_only=True))
        z = torch.randn(2, 5, 3)
        self.assertTrue(torch.allclose(ple(z), restored(z)))
        edges.add_(10)
        self.assertFalse(torch.allclose(restored.edges, edges))


class RevINModelTests(unittest.TestCase):
    def test_norm_matches_vendor_and_time_axis(self):
        torch.manual_seed(3)
        x = torch.randn(4, 9, 3)
        vendor = RevIN(3, eps=1e-5, affine=False)
        vendor._get_statistics(x)
        z_v = (x - vendor.mean) / vendor.stdev
        z, mean, std = instance_norm(x, eps=1e-5)
        self.assertTrue(torch.allclose(z, z_v, atol=1e-6))
        self.assertTrue(torch.allclose(mean, x.mean(dim=1, keepdim=True), atol=1e-6))
        # Not a batch statistic and not a channel-mixed statistic.
        self.assertFalse(torch.allclose(mean, x.mean(dim=0, keepdim=True)))
        self.assertEqual(tuple(mean.shape), (4, 1, 3))
        y = torch.randn(4, 5, 3)
        vendor_aff = RevIN(3, eps=1e-5, affine=True)
        with torch.no_grad():
            vendor_aff.affine_weight.normal_()
            vendor_aff.affine_bias.normal_()
        vendor_aff._get_statistics(x)
        ref = vendor_aff._denormalize(y.clone())
        got = denorm_channel(
            y[:, :, 1],
            vendor_aff.mean,
            vendor_aff.stdev,
            vendor_aff.affine_weight,
            vendor_aff.affine_bias,
            1,
            vendor_aff.eps,
        )
        self.assertTrue(torch.allclose(got, ref[:, :, 1], atol=1e-6))

    def _models(self, task, lookback, horizon, n_features, cnt_index=0):
        torch.manual_seed(4)
        edges = fit_revin_quantile_edges(torch.randn(8, lookback, n_features))
        plain = build_model(
            "PatchTST",
            task=task,
            n_features=n_features,
            lookback=lookback,
            horizon=horizon,
            cnt_index=cnt_index,
        )
        ple = build_model(
            "PatchTST-PLE-quantile",
            task=task,
            n_features=n_features,
            lookback=lookback,
            horizon=horizon,
            edges=edges,
            cnt_index=cnt_index,
        )
        return plain, ple

    def test_classification_does_not_denorm_forecast_does(self):
        plain, ple = self._models("classification", 60, 1, 8)
        x = torch.randn(3, 60, 8)
        x2 = x.clone()
        x2[..., 0] = x[..., 0] * 10 + 3
        for model in (plain, ple):
            model.eval()
            self.assertTrue(torch.allclose(model(x), model(x2), atol=1e-4, rtol=1e-4))
        f_plain, f_ple = self._models("forecast", 60, 4, 8, cnt_index=0)
        # Use lookback 60 with horizon 4: the mixing head does not depend on a
        # real 168-hour series. Channel 0 is the denorm channel.
        x = torch.randn(3, 60, 8)
        x[:, :, 0] += torch.linspace(0, 1, 60)
        x2 = x.clone()
        x2[..., 0] = x[..., 0] * 10 + 3
        for model in (f_plain, f_ple):
            model.eval()
            y1 = model(x)
            y2 = model(x2)
            self.assertEqual(tuple(y1.shape), (3, 4))
            self.assertTrue(torch.allclose(y2, y1 * 10 + 3, atol=1e-4, rtol=1e-4))

    def test_pre_affine_ignores_affine_and_stats_follow_the_batch(self):
        model, _ = self._models("forecast", 32, 2, 4)
        # Rebuild a short lookback by the helper above (32). Check stats.
        x1 = torch.randn(2, 32, 4)
        x2 = torch.randn(2, 32, 4) + 4
        model.eval()
        z1 = model.pre_affine(x1).detach().clone()
        with torch.no_grad():
            model.revin.affine_weight.mul_(3)
            model.revin.affine_bias.add_(2)
        z2 = model.pre_affine(x1)
        self.assertTrue(torch.allclose(z1, z2, atol=1e-6))
        model.pre_affine(x2)
        self.assertTrue(torch.allclose(model.revin.mean, x2.mean(1, keepdim=True), atol=1e-6))
        self.assertFalse(torch.allclose(model.revin.mean, x1.mean(1, keepdim=True), atol=1e-4))

    def test_eval_freezes_batchnorm_and_edges(self):
        torch.manual_seed(5)
        edges = fit_revin_quantile_edges(torch.randn(4, 60, 8))
        model = build_model(
            "PatchTST-PLE-quantile",
            task="classification",
            n_features=8,
            lookback=60,
            horizon=1,
            dropout=0.0,
            edges=edges,
        )
        # ``norm='LayerNorm'`` is accepted by the backbone and not forwarded.
        # The encoder that actually runs is BatchNorm1d, same as run_detect.Model.
        norm = model.net.backbone.encoder.layers[0].norm_attn
        self.assertIsInstance(norm[1], torch.nn.BatchNorm1d)
        x = torch.randn(6, 60, 8)
        model.train()
        running = norm[1].running_mean.detach().clone()
        model(x)
        self.assertFalse(torch.equal(norm[1].running_mean, running))
        model.eval()
        frozen = norm[1].running_mean.detach().clone()
        b = model(x).detach()
        c = model(x).detach()
        self.assertTrue(torch.equal(norm[1].running_mean, frozen))
        self.assertTrue(torch.equal(b, c))
        probs = predict(model, x, task="classification")
        self.assertFalse(model.training)
        self.assertEqual(tuple(probs.shape), (6,))

    def test_zero_ple_matches_plain_and_param_delta(self):
        torch.manual_seed(6)
        edges = torch.linspace(-2, 2, N_BINS + 1).repeat(8, 1)
        plain = build_model("PatchTST", task="classification", n_features=8, lookback=60)
        ple = build_model(
            "PatchTST-PLE-quantile",
            task="classification",
            n_features=8,
            lookback=60,
            edges=edges,
        )
        copy_shared_weights(plain, ple)
        with torch.no_grad():
            ple.ple.W.zero_()
        x = torch.randn(2, 60, 8)
        plain.eval()
        ple.eval()
        self.assertTrue(torch.allclose(plain(x), ple(x), atol=1e-5))
        with torch.no_grad():
            ple.ple.W.add_(0.3)
        self.assertFalse(torch.allclose(plain(x), ple(x), atol=1e-4))
        old = run_detect.Model("PatchTST", 0.1)
        n_old = sum(p.numel() for p in old.parameters())
        n_new = sum(p.numel() for p in plain.parameters())
        n_ple = sum(p.numel() for p in ple.parameters())
        self.assertEqual(n_new - n_old, 16)  # RevIN affine only
        self.assertEqual(n_ple - n_new, 8 * N_BINS)
        self.assertEqual(plain.n_patches, 15)
        tabm = build_model("TabM", task="classification", n_features=8, lookback=60)
        self.assertEqual(type(tabm.net).__name__, "TabM")
        self.assertTrue(type(tabm.net).__module__.startswith("tabm"))
        self.assertEqual(sum(p.numel() for p in tabm.parameters()), 48656)
        bike_plain = build_model("PatchTST", task="forecast", n_features=13, lookback=168, horizon=24)
        bike_ple = build_model(
            "PatchTST-PLE-quantile",
            task="forecast",
            n_features=13,
            lookback=168,
            horizon=24,
            edges=torch.linspace(-2, 2, N_BINS + 1).repeat(13, 1),
        )
        self.assertEqual(
            sum(p.numel() for p in bike_ple.parameters()) - sum(p.numel() for p in bike_plain.parameters()),
            13 * N_BINS,
        )
        self.assertEqual(bike_plain.n_patches, 42)

    def test_last_singleton_batch_is_merged(self):
        order = torch.arange(257)
        batches = iter_batches(order, 256)
        self.assertEqual([int(b.numel()) for b in batches], [257])
        exact = iter_batches(torch.arange(512), 256)
        self.assertEqual([int(b.numel()) for b in exact], [256, 256])

    def test_training_step_does_not_move_edges(self):
        torch.manual_seed(7)
        lookback, n, c = 60, 40, 8
        x = torch.randn(n, lookback, c).numpy()
        y = np.array([0, 1] * (n // 2), dtype=np.float32)
        edges = fit_revin_quantile_edges(torch.as_tensor(x[:30]))
        before = edges.clone()

        def factory():
            return build_model(
                "PatchTST-PLE-quantile",
                task="classification",
                n_features=c,
                lookback=lookback,
                edges=edges.clone(),
            )

        scores = iter([0.2, 0.1])

        def score_from_pred(_prob):
            return next(scores)

        out = train_one(
            factory,
            task="classification",
            x_train=x[:30],
            y_train=y[:30],
            x_val=x[30:],
            y_val=y[30:],
            score_from_pred=score_from_pred,
            maximize=True,
            seed=0,
            max_epochs=4,
            patience=1,
            batch_size=16,
        )
        self.assertEqual(out["epochs_ran"], 2)
        self.assertEqual(out["best_epoch"], 1)
        self.assertTrue(torch.equal(out["model"].ple.edges, before))
        self.assertTrue(all(not p.requires_grad for p in [out["model"].ple.edges]))
        self.assertIsNone(out["model"].ple.edges.grad)


class SplitAndWindowTests(unittest.TestCase):
    def test_groups_and_old_equal_bin_stay(self):
        self.assertEqual(groups_are_intact(), [])
        spec = load_splits()
        names = set()
        for fold in spec["folds"]:
            for split in ("train", "val", "test"):
                names.update(fold[split])
        self.assertTrue(all(not n.startswith("anomaly-free") for n in names))
        self.assertEqual(len(names), 34)
        src = (EXP / "run_detect.py").read_text()
        self.assertIn("revin=False", src)
        self.assertIn("linspace(-2,2", src)
        with self.assertRaises(ValueError):
            build_model(EQUAL_BIN_MODEL, task="classification", n_features=8, lookback=60)
        jobs = plan_jobs()
        self.assertEqual(len(jobs), 18)
        self.assertTrue(all(job.seed == 0 for job in jobs[:9]))
        self.assertEqual(jobs[9].seed, 1)
        self.assertEqual({job.model for job in jobs}, set(PILOT_MODELS))
        self.assertEqual(len(plan_jobs(seeds=(0,))), 9)
        with self.assertRaises(SystemExit):
            assert_pilot_size(plan_jobs(seeds=(0, 1, 2), skab_folds=(0, 1, 2, 3, 4)))

    def test_skab_grid_fill_and_label(self):
        start = pd.Timestamp("2020-01-01")
        # Missing t=2s is a one-step hole and must be filled from the past.
        times = [start + pd.Timedelta(seconds=s) for s in (0, 1, 3, 4, 5, 6, 7)]
        values = np.linspace(1, 7, len(times), dtype=np.float32)
        frame = pd.DataFrame({"datetime": times, "anomaly": [0, 1, 0, 1, 0, 1, 0], "changepoint": [1, 0, 1, 0, 1, 0, 1]})
        for feature in FEATURES:
            frame[feature] = values
        x, y = windows_from_frame(frame, lookback=4, stride=1)
        self.assertGreater(len(y), 0)
        # Grid second 2 is filled with the value at second 1 (index 1 of `values`), not the future.
        filled = x[0, 2, 0].item()
        self.assertAlmostEqual(filled, float(values[1]), places=5)
        frame["changepoint"] = 1 - frame["changepoint"]
        x2, y2 = windows_from_frame(frame, lookback=4, stride=1)
        self.assertTrue(np.array_equal(y, y2))
        self.assertTrue(np.allclose(x, x2))
        bare = frame.drop(columns=["anomaly", "changepoint"])
        empty_x, empty_y = windows_from_frame(bare, lookback=4, stride=1)
        self.assertEqual(len(empty_y), 0)
        # Two-second hole leaves a NaN, so a window covering it is absent.
        times2 = [start + pd.Timedelta(seconds=s) for s in range(0, 12) if s not in (5, 6)]
        frame2 = pd.DataFrame(
            {
                "datetime": times2,
                "anomaly": np.zeros(len(times2)),
                "changepoint": np.zeros(len(times2)),
            }
        )
        for feature in FEATURES:
            frame2[feature] = 1.0
        _, y_gap = windows_from_frame(frame2, lookback=8, stride=1)
        _, y_full = windows_from_frame(
            pd.DataFrame(
                {
                    "datetime": [start + pd.Timedelta(seconds=s) for s in range(12)],
                    "anomaly": np.zeros(12),
                    **{feature: np.ones(12) for feature in FEATURES},
                }
            ),
            lookback=8,
            stride=1,
        )
        self.assertLess(len(y_gap), len(y_full))

    def test_bike_windows_splits_and_exclusions(self):
        self.assertEqual(BIKE_LOOKBACK, 168)
        self.assertEqual(BIKE_HORIZON, 24)
        self.assertNotIn("casual", BIKE_FEATURES)
        self.assertNotIn("registered", BIKE_FEATURES)
        self.assertNotIn("instant", BIKE_FEATURES)
        frame = _bike_frame(90, gap_at={40}, wind_zero_at={10})
        casual = frame["casual"].copy()
        built = build_bike_dataset(frame, lookback=6, horizon=3)
        frame["casual"] = casual + 50
        frame["registered"] = frame["cnt"] - frame["casual"]
        rebuilt = build_bike_dataset(frame, lookback=6, horizon=3)
        self.assertTrue(np.allclose(built["parts"]["train"]["x"], rebuilt["parts"]["train"]["x"]))
        finite_cnt = built["raw"][np.isfinite(built["raw"][:, 0]), 0]
        self.assertTrue(np.all(finite_cnt > 0))
        self.assertTrue(np.isnan(built["raw"][40]).all())
        self.assertEqual(built["report"]["n_missing_hours"], 1)
        report = built["report"]
        self.assertEqual(
            report["n_kept"] + report["n_dropped_gap"] + report["dropped_cross_split"],
            report["n_candidate"],
        )
        self.assertGreater(report["n_dropped_gap"], 0)
        # Recorded windspeed 0 is not a missing hour.
        self.assertGreater(report["windspeed_recorded_zeros"], 0)
        self.assertGreater(report["splits"]["train"]["kept"], 0)
        scaled = built["scaled"]
        for split in ("train", "val", "test"):
            part = built["parts"][split]
            for row_i, (row, i) in enumerate(zip(part["x"], part["target_index"])):
                self.assertTrue(np.allclose(row, scaled[i - 6 : i], atol=1e-6))
                self.assertTrue(np.allclose(part["y_raw"][row_i], built["raw"][i : i + 3, 0]))
                self.assertLess(int(i) - 6, int(i))
        # No target hour sits on the other side of a cut.
        for split, lo, hi in (
            ("train", 0, built["cut1"]),
            ("val", built["cut1"], built["cut2"]),
            ("test", built["cut2"], built["report"]["n_grid"]),
        ):
            for i in built["parts"][split]["target_index"]:
                self.assertGreaterEqual(i, lo)
                self.assertLess(i + 3 - 1, hi)
        self.assertGreater(built["report"]["splits"]["val"]["context_includes_earlier_split"], 0)
        # Scaler ignores a huge test-period edit.
        edited = frame.copy()
        # Last rows are the test period.
        edited.loc[edited.index[-1], "temp"] = 1e6
        edited_built = build_bike_dataset(edited, lookback=6, horizon=3)
        self.assertTrue(np.allclose(built["scaler_mean"], edited_built["scaler_mean"]))
        naive = seasonal_naive(built, period=2, split="test")
        self.assertGreater(naive["n"], 0)
        self.assertTrue(np.isfinite(naive["MAE"]))


class OfficialDataTests(unittest.TestCase):
    def test_official_files_when_present(self):
        hour = EXP / "data" / "bike" / "hour.csv"
        skab = EXP / "data" / "valve1" / "0.csv"
        if not skab.exists() or not hour.exists():
            self.skipTest("official files not downloaded")
        import hashlib
        import json

        manifest = json.loads((EXP / "data_hashes.json").read_text())
        for rel, digest in manifest.items():
            got = hashlib.sha256((EXP / rel).read_bytes()).hexdigest()
            self.assertEqual(got, digest, rel)
        pins = json.loads((EXP / "quantile_pilot" / "data_pins.json").read_text())
        self.assertEqual(hashlib.sha256(hour.read_bytes()).hexdigest(), pins["bike"]["hour_csv_sha256"])
        loaded = load_skab_fold(0)
        self.assertEqual(len(loaded["parts"]["train"]["y"]), 5067)
        self.assertEqual(len(loaded["parts"]["val"]["y"]), 1837)
        self.assertEqual(len(loaded["parts"]["test"]["y"]), 1771)
        # Same windows as the historical pipeline, not a new row split.
        spec = load_splits()["folds"][0]
        frames, raw = run_detect.prepare(spec["train"] + spec["val"] + spec["test"])
        items = [z for n in spec["train"] for z in raw[n]]
        x = np.stack([z[0] for z in items])
        y = np.asarray([z[1] for z in items], np.float32)
        mean = loaded["scaler_mean"]
        scale = loaded["scaler_scale"]
        x_s = ((x - mean) / scale).astype("float32")
        self.assertTrue(np.allclose(x_s, loaded["parts"]["train"]["x"]))
        self.assertTrue(np.array_equal(y, loaded["parts"]["train"]["y"]))
        free = pd.read_csv(EXP / "data" / "anomaly-free" / "anomaly-free.csv", sep=";")
        self.assertNotIn("anomaly", free.columns)
        self.assertEqual(len(windows_from_frame(free)[1]), 0)


if __name__ == "__main__":
    unittest.main()
