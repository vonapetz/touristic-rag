#!/usr/bin/env python3
"""Скачивание датасета."""

import sys
sys.path.insert(0, ".")

from app.data_processor import download_data

if __name__ == "__main__":
    path = download_data()
    print(f"Данные скачаны: {path}")