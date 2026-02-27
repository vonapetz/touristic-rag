"""
Очистка данных и создание документов.
Полный pipeline: загрузка → фильтрация → TF-IDF → дедупликация → обогащение.
"""

import os
import re
from typing import List

import pandas as pd
import numpy as np
import gdown
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings
from app.logger import get_logger

logger = get_logger("data")


def download_data() -> str:
    """Скачивает датасет если его нет."""
    os.makedirs(os.path.dirname(settings.data_path), exist_ok=True)

    if not os.path.exists(settings.data_path):
        logger.info("downloading_data", url=settings.data_url, path=settings.data_path)
        gdown.download(settings.data_url, settings.data_path, quiet=False)

    return settings.data_path


def load_and_clean_data() -> pd.DataFrame:
    """Полный pipeline очистки данных."""

    path = download_data()
    data = pd.read_csv(path)
    logger.info("data_loaded", rows=len(data), columns=list(data.columns))

    # Определяем реальные имена колонок (регистронезависимо)
    col_map = {c.lower(): c for c in data.columns}
    name_col = col_map.get("name", "Name")
    city_col = col_map.get("city", "City")
    desc_col = col_map.get("description", "description")
    # Переименовываем к стандартным именам
    rename = {}
    if name_col != "Name":
        rename[name_col] = "Name"
    if city_col != "City":
        rename[city_col] = "City"
    if desc_col != "description":
        rename[desc_col] = "description"
    if rename:
        data = data.rename(columns=rename)

    text_col = "description"
    has_en = "en_txt" in data.columns
    tfidf_col = "en_txt" if has_en else "description"

    # Базовая фильтрация
    clean = data.copy()
    clean = clean[clean[text_col].notna()].copy()
    clean = clean[clean[text_col].str.len() >= settings.min_description_length].copy()
    if "Name" in clean.columns:
        clean = clean[clean["Name"].notna() & (clean["Name"].str.len() >= 3)].copy()
    else:
        clean["Name"] = "Неизвестно"
    if "City" in clean.columns:
        clean = clean[clean["City"].notna()].copy()
    else:
        clean["City"] = "Неизвестный город"
    logger.info("basic_filter", remaining=len(clean))

    # TF-IDF фильтрация (смягчена: min_df=3, max_df=0.7)
    tfidf_texts = clean[tfidf_col].fillna("").astype(str)
    vp = {
        "min_df": settings.tfidf_min_df,
        "max_df": settings.tfidf_max_df,
        "ngram_range": (1, 2),
    }
    if has_en:
        vp["stop_words"] = "english"

    vectorizer = TfidfVectorizer(**vp)
    X = vectorizer.fit_transform(tfidf_texts)
    token_scores = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))

    suspicious = set(
        t for t, s in token_scores.items()
        if s < settings.tfidf_suspicious_low or s > settings.tfidf_suspicious_high
    )

    clean = clean[
        ~clean[tfidf_col].apply(
            lambda t: bool(set(str(t).lower().split()) & suspicious) if pd.notna(t) else False
        )
    ].copy()
    logger.info("tfidf_filter", remaining=len(clean), suspicious_tokens=len(suspicious))

    # Дедупликация
    # Защитные проверки: добавляем отсутствующие опциональные колонки
    for col in ["image", "WikiData", "Lon", "Lat"]:
        if col not in clean.columns:
            clean[col] = np.nan

    clean["has_image"] = clean["image"].apply(
        lambda x: isinstance(x, str) and len(str(x)) > 100
    )

    def select_best(group):
        scores = pd.Series(0.0, index=group.index)
        desc_lens = group[text_col].str.len().fillna(0)
        max_len = desc_lens.max()
        if max_len > 0:
            scores += (desc_lens / max_len) * 5
        scores += group["WikiData"].notna().astype(float) * 2
        scores += group["has_image"].astype(float) * 3
        scores += (group["Lon"].notna() & group["Lat"].notna()).astype(float)
        # Двойные скобки возвращают DataFrame (а не Series),
        # чтобы pandas не переносил Name/City в индекс при apply
        return group.loc[[scores.idxmax()]]

    final = clean.groupby(
        ["Name", "City"], group_keys=False
    ).apply(select_best).reset_index(drop=True)
    logger.info("deduplicated", remaining=len(final))

    # Обогащение
    def rich_text(row):
        parts = [f"{str(row['Name']).strip()}, город {str(row['City']).strip()}."]
        desc = str(row[text_col]).strip()
        if desc and desc != "nan":
            parts.append(desc)
        return " ".join(parts)

    final["rich_text"] = final.apply(rich_text, axis=1)

    # Сохраняем
    os.makedirs(os.path.dirname(settings.processed_data_path), exist_ok=True)
    final.to_csv(settings.processed_data_path, index=False)

    logger.info(
        "data_processed",
        original=len(data),
        final=len(final),
        cities=int(final["City"].nunique()),
        places=int(final["Name"].nunique()),
    )

    return final


def create_documents(df: pd.DataFrame) -> List[Document]:
    """Создаёт LangChain Document из DataFrame."""

    documents = []

    for _, row in df.iterrows():
        text = str(row.get("rich_text", "")).strip()
        if not text or text == "nan" or len(text) < 20:
            continue

        metadata = {
            "name": str(row["Name"]) if pd.notna(row.get("Name")) else "Неизвестно",
            "city": str(row["City"]) if pd.notna(row.get("City")) else "Неизвестный город",
        }

        # FIX: валидация координат — только допустимые значения
        if pd.notna(row.get("Lat")) and pd.notna(row.get("Lon")):
            try:
                lat = float(row["Lat"])
                lon = float(row["Lon"])
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    metadata["lat"] = lat
                    metadata["lon"] = lon
            except (ValueError, TypeError):
                pass

        if pd.notna(row.get("WikiData")):
            metadata["wikidata"] = str(row["WikiData"])

        # FIX: полный URL изображения — не обрезать до 100 символов
        if row.get("has_image", False) and isinstance(row.get("image"), str):
            metadata["image"] = str(row["image"])

        # Чистый текст без prefix — prefix добавляет E5EmbeddingsWrapper
        documents.append(Document(page_content=text, metadata=metadata))

    logger.info("documents_created", count=len(documents))
    return documents
