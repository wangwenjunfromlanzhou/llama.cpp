#!/usr/bin/env python3
"""Failing tests for DSpark Gemma4 layer-structure tensors.

These assert that conversion does NOT drop the Gemma4-specific tensors that the
DSpark decoder graph needs: layer_scalar, post_attention_layernorm,
post_feedforward_layernorm. They run against the classmethod filter_tensors of
Gemma4DSparkModel, which only inspects tensor names -- no model instantiation
required.

Run: python -m unittest tests.test_dspark_gemma4_tensors
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conversion.gemma import Gemma4DSparkModel  # noqa: E402


def _filter(name):
    """Return the (possibly renamed) name retained by the filter, or None."""
    item = (name, None)
    out = Gemma4DSparkModel.filter_tensors(item)
    return None if out is None else out[0]


class TestDSParkGemma4LayerTensors(unittest.TestCase):
    def test_layer_scalar_is_kept(self):
        self.assertIsNotNone(
            _filter("model.layers.0.layer_scalar"),
            "layer_scalar is dropped by conversion but the DSpark decoder graph must multiply each layer output by it",
        )

    def test_post_attention_layernorm_is_kept(self):
        self.assertIsNotNone(
            _filter("model.layers.0.post_attention_layernorm.weight"),
            "post_attention_layernorm.weight is dropped but DSpark Gemma4 needs the sandwich-norm path",
        )

    def test_post_feedforward_layernorm_is_kept(self):
        self.assertIsNotNone(
            _filter("model.layers.0.post_feedforward_layernorm.weight"),
            "post_feedforward_layernorm.weight is dropped but DSpark Gemma4 needs the sandwich-norm path",
        )

    def test_v_proj_passes_filter(self):
        # filter_tensors must let model.layers.N.self_attn.v_proj.weight through,
        # even though Gemma4 k_eq_v layers ship only q+k in the checkpoint --
        # modify_tensors duplicates k -> v on the fly.
        self.assertIsNotNone(_filter("model.layers.0.self_attn.v_proj.weight"))


if __name__ == "__main__":
    unittest.main()
