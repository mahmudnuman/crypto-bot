"""
tests/test_downloader.py — Unit tests for the data downloader.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDownloadRest(unittest.TestCase):

    @patch("data.downloader.requests.get")
    def test_basic_pagination(self, mock_get):
        """Verifies pagination stops when fewer than 1000 rows are returned."""
        def make_row(ts_ms):
            return [ts_ms, "1.0", "1.1", "0.9", "1.05",
                    "100", ts_ms + 299_999, "105", "50", "40", "42", "0"]

        page1 = [make_row(k * 300_000) for k in range(1000)]
        page2 = [make_row((1000 + k) * 300_000) for k in range(300)]

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.side_effect = [page1, page2]

        from data.downloader import download_rest
        df = download_rest("BTCUSDT", "1h", 0, 9_999_999_999_999)
        self.assertEqual(len(df), 1300)

    @patch("data.downloader.requests.get")
    def test_rate_limit_backoff(self, mock_get):
        """429 followed by 200 should succeed after backoff."""
        mock_429 = MagicMock()
        mock_429.status_code = 429

        row = [1000, "1.0", "1.1", "0.9", "1.05", "100",
               1299999, "105", "50", "40", "42", "0"]
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = [row] * 10

        mock_get.side_effect = [mock_429, mock_200]
        from data.downloader import download_rest
        # Should not raise
        with patch("time.sleep"):
            df = download_rest("BTCUSDT", "1h", 0, 9_999_999_999_999)
        self.assertGreater(len(df), 0)

    def test_empty_response(self):
        """Empty API response should return empty DataFrame."""
        with patch("data.downloader.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = []
            from data.downloader import download_rest
            df = download_rest("DOGEUSDT", "1h", 0, 9_999_999_999_999)
        self.assertTrue(df.empty)


class TestStore(unittest.TestCase):

    def test_upsert_dedup(self):
        """Saving the same rows twice should not create duplicates."""
        import tempfile
        import os
        from unittest.mock import patch as mp

        with tempfile.TemporaryDirectory() as tmpdir:
            with mp("data.store.KLINES_DIR", Path(tmpdir)):
                from data import store
                store.KLINES_DIR = Path(tmpdir)

                # Build a tiny synthetic DataFrame
                df = pd.DataFrame({
                    "open_time":  [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 00:05")],
                    "open":       [100.0, 101.0],
                    "high":       [102.0, 103.0],
                    "low":        [99.0,  100.0],
                    "close":      [101.0, 102.0],
                    "volume":     [500.0, 600.0],
                    "close_time": [pd.Timestamp("2024-01-01 00:04:59"),
                                   pd.Timestamp("2024-01-01 00:09:59")],
                    "quote_volume":         [50000.0, 61000.0],
                    "num_trades":           [200,     210],
                    "taker_buy_base_vol":   [250.0,   300.0],
                    "taker_buy_quote_vol":  [25000.0, 30500.0],
                    "ignore":               ["0", "0"],
                })

                store.save("TESTUSDT", "1h", df)
                store.save("TESTUSDT", "1h", df)   # duplicate save
                loaded = store.load("TESTUSDT", "1h")
                self.assertEqual(len(loaded), 2, "Dedup failed — got duplicates")


if __name__ == "__main__":
    unittest.main()
