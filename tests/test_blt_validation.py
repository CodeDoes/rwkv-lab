"""Unit and integration tests for the BLT comparative validation pipeline."""

from __future__ import annotations

import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

import json
from pathlib import Path
import pytest
import torch

from rwkv_lab.blt_validation import (
    StandardRWKV7LanguageModel,
    run_comparative_validation,
)


def test_standard_rwkv7_model_construction():
    """Verify that StandardRWKV7LanguageModel builds and performs forward passes correctly."""
    torch.manual_seed(42)
    vocab_size = 256
    d_model = 32
    n_layers = 2

    model = StandardRWKV7LanguageModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)

    # Check that parameters are initialized and not NaN
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"Parameter {name} contains NaN!"

    # Input batch size 2, sequence length 16
    ids = torch.randint(0, vocab_size, (2, 16))
    logits = model(ids)

    assert logits.shape == (2, 16, vocab_size), f"Expected shape (2, 16, 256), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Forward pass logits contain NaN!"


def test_comparative_validation_pipeline():
    """Verify that run_comparative_validation runs successfully, verifies metrics, and writes JSON report."""
    report_path = Path("runs/blt_comparison_report.json")
    # Clean up existing report
    report_path.unlink(missing_ok=True)

    # Run comparative validation for 5 epochs (to keep test suite fast)
    report = run_comparative_validation(epochs=5, lr=3e-3, threshold=3.5, max_patch=16)

    # 1. Assert structure of return dictionary
    assert "parameters" in report
    assert "training" in report
    assert "benchmark" in report
    assert "dynamic_adaptation" in report

    # 2. Assert values in parameters section
    assert report["parameters"]["standard_rwkv7"] > 0
    assert report["parameters"]["blt_rwkv7"] > 0
    # BLT has more parameters due to the entropy head
    assert report["parameters"]["blt_rwkv7"] > report["parameters"]["standard_rwkv7"]

    # 3. Assert values in training section
    assert 0.0 < report["training"]["standard_final_train_loss"] < 10.0
    assert 0.0 < report["training"]["standard_val_loss"] < 10.0
    assert 0.0 < report["training"]["blt_final_train_loss"] < 10.0
    assert 0.0 < report["training"]["blt_val_loss"] < 10.0

    # 4. Assert values in benchmark section
    assert report["benchmark"]["sequence_shape"] == [4, 256]
    assert report["benchmark"]["standard_latency_ms"] > 0.0
    assert report["benchmark"]["blt_latency_ms"] > 0.0
    assert report["benchmark"]["standard_throughput_bps"] > 0.0
    assert report["benchmark"]["blt_throughput_bps"] > 0.0
    assert report["benchmark"]["speedup"] > 0.0

    # 5. Assert values in dynamic_adaptation section
    assert report["dynamic_adaptation"]["initial_entropy"] > 0.0
    assert report["dynamic_adaptation"]["final_entropy"] > 0.0
    assert report["dynamic_adaptation"]["initial_patch_length"] >= 1.0
    assert report["dynamic_adaptation"]["final_patch_length"] >= 1.0
    assert isinstance(report["dynamic_adaptation"]["successful"], bool)

    # 6. Assert JSON report file is written and matches returned report
    assert report_path.is_file(), "Report file was not written!"
    with report_path.open() as f:
        saved_report = json.load(f)

    assert saved_report == report
