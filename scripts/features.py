#!/usr/bin/env python3
"""统一特征入口：surface 风格特征（style_features）+ 语句逻辑/语气特征（logic_features）。"""
import numpy as np
import style_features as S
import logic_features as L

KINDS = ["style", "style+logic"]


def names(kind="style+logic"):
    return list(S.FEATURE_NAMES) + (list(L.NAMES) if kind == "style+logic" else [])


def extract(text, kind="style+logic"):
    v = S.extract(text)
    if kind == "style+logic":
        v = np.concatenate([v, L.extract(text)])
    return v.astype(np.float32)


def extract_many(texts, kind="style+logic"):
    return np.stack([extract(t, kind) for t in texts])
