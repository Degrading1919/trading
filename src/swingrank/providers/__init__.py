"""Market-data provider interfaces and reference adapters."""

from swingrank.providers.base import MarketDataProvider
from swingrank.providers.csv import CsvDataProvider
from swingrank.providers.memory import InMemoryDataProvider

__all__ = ["CsvDataProvider", "InMemoryDataProvider", "MarketDataProvider"]
