"""Tests for the market data parser."""

from datetime import date

import pytest

from weather_bot.market import _extract_city_key, _extract_target_date, _parse_temperature_bucket


class TestParseBucket:
    def test_single_celsius(self):
        bucket = _parse_temperature_bucket(
            "Will the highest temperature in London be 13°C on March 20?",
            "13°C",
            "celsius",
        )
        assert bucket is not None
        assert bucket.lower == 12.5
        assert bucket.upper == 13.5
        assert not bucket.is_lower_tail
        assert not bucket.is_upper_tail

    def test_range_fahrenheit(self):
        bucket = _parse_temperature_bucket(
            "Will the highest temperature in NYC be 58-59°F?",
            "58-59",
            "fahrenheit",
        )
        assert bucket is not None
        assert bucket.lower == 57.5
        assert bucket.upper == 59.5

    def test_lower_tail(self):
        bucket = _parse_temperature_bucket(
            "Will it be 27°F or below?",
            "27 or below",
            "fahrenheit",
        )
        assert bucket is not None
        assert bucket.is_lower_tail
        assert bucket.upper == 27.5

    def test_upper_tail(self):
        bucket = _parse_temperature_bucket(
            "Will it be 38 or above?",
            "≥38",
            "fahrenheit",
        )
        assert bucket is not None
        assert bucket.is_upper_tail
        assert bucket.lower == 37.5

    def test_upper_tail_plus(self):
        bucket = _parse_temperature_bucket(
            "Will it be 19+?",
            "19+",
            "celsius",
        )
        assert bucket is not None
        assert bucket.is_upper_tail


class TestExtractCity:
    def test_nyc(self):
        assert _extract_city_key("NYC Daily High Temperature March 20") == "nyc"

    def test_london(self):
        assert _extract_city_key("London High Temperature") == "london"

    def test_unknown(self):
        assert _extract_city_key("Random Event Title") is None


class TestExtractDate:
    def test_march_20(self):
        d = _extract_target_date("", "Will the highest temperature in London be 13°C on March 20?")
        assert d is not None
        assert d.month == 3
        assert d.day == 20

    def test_with_year(self):
        d = _extract_target_date("NYC temp March 20, 2026", "")
        assert d is not None
        assert d == date(2026, 3, 20)

    def test_no_date(self):
        d = _extract_target_date("Some event", "Some question")
        assert d is None
