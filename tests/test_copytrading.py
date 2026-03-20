"""Tests for the copy-trading module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from weather_bot.copytrading import (
    TrackedWallet,
    WalletPosition,
    add_wallet,
    load_wallets,
    remove_wallet,
    save_wallets,
    summarize_positions,
    toggle_wallet,
)


@pytest.fixture(autouse=True)
def tmp_wallets_file(tmp_path):
    """Redirect wallets file to a temporary location."""
    wallets_file = tmp_path / "wallets.json"
    with patch("weather_bot.copytrading.WALLETS_FILE", wallets_file):
        yield wallets_file


class TestWalletManagement:
    def test_add_wallet(self):
        w = add_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11", label="whale1")
        assert w.address == "0x594edb9112f526fa6a80b8f858a6379c8a2c1c11"
        assert w.label == "whale1"
        assert w.enabled is True

    def test_add_wallet_normalizes_case(self):
        w = add_wallet("0x594EDB9112F526FA6A80B8F858A6379C8A2C1C11")
        assert w.address == "0x594edb9112f526fa6a80b8f858a6379c8a2c1c11"

    def test_add_duplicate_wallet(self):
        add_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11", label="first")
        w = add_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11", label="second")
        # Should not duplicate, but should update label if empty
        wallets = load_wallets()
        assert len(wallets) == 1

    def test_add_invalid_address(self):
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            add_wallet("not-an-address")

    def test_remove_wallet(self):
        add_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11")
        assert remove_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11") is True
        assert load_wallets() == []

    def test_remove_nonexistent(self):
        assert remove_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11") is False

    def test_toggle_wallet(self):
        add_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11")
        assert toggle_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11", enabled=False) is True
        wallets = load_wallets()
        assert wallets[0].enabled is False

    def test_load_empty(self):
        assert load_wallets() == []

    def test_persistence(self, tmp_wallets_file):
        add_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11", label="test")
        # Verify file was written
        data = json.loads(tmp_wallets_file.read_text())
        assert len(data) == 1
        assert data[0]["address"] == "0x594edb9112f526fa6a80b8f858a6379c8a2c1c11"

    def test_multiple_wallets(self):
        add_wallet("0x594edb9112f526fa6a80b8f858a6379c8a2c1c11", label="w1")
        add_wallet("0x87650b9f63563f7c456d9bbcceee5f9faf06ed81", label="w2")
        add_wallet("0x5f211a24da4c005d9438a1ea269673b85ed0b376", label="w3")
        wallets = load_wallets()
        assert len(wallets) == 3


class TestSummarizePositions:
    def test_consensus_detection(self):
        positions = {
            "0xaaa0000000000000000000000000000000000001": [
                WalletPosition(
                    wallet="0xaaa0000000000000000000000000000000000001",
                    market_slug="will-it-rain",
                    title="Will it rain tomorrow?",
                    outcome="Yes",
                    size=100.0,
                    avg_price=0.60,
                    current_price=0.65,
                    condition_id="cond123",
                ),
            ],
            "0xbbb0000000000000000000000000000000000002": [
                WalletPosition(
                    wallet="0xbbb0000000000000000000000000000000000002",
                    market_slug="will-it-rain",
                    title="Will it rain tomorrow?",
                    outcome="Yes",
                    size=50.0,
                    avg_price=0.55,
                    current_price=0.65,
                    condition_id="cond123",
                ),
            ],
        }
        summary = summarize_positions(positions)
        assert len(summary) == 1
        assert summary[0]["wallet_count"] == 2
        assert summary[0]["total_size"] == 150.0

    def test_empty_positions(self):
        assert summarize_positions({}) == []
