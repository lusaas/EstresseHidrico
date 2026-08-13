#!/usr/bin/env python3
"""Helpers compartilhados da validacao cross-learning entre genotipos.

Contem o que os tres scripts de `group_kfold` usam:

- carregamento dos dados (`carregar`) e das estatisticas de estresse por
  genotipo x dia (`comparacao_estresse.csv`);
- a selecao das top-K bandas de um genotipo num dia (representantes
  Spearman + significancia q_FDR<=0,05 + VIP do PLS-DA), identica a regua de
  `analiseBR16PorDia/spearman_plsda_br16_manha.py`;
- o recorte do espectro nas bandas selecionadas;
- os modelos (PLS-DA e Random Forest) e as metricas de classificacao,
  incluindo balanced accuracy.

Nenhuma funcao aqui depende dos genotipos de teste de um fold: a selecao
sempre usa apenas o material--dia de interesse, para nao vazar informacao.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

for sub in (
    "testeDeNormalidade",
    "reducaoColinearidade",
    "analiseBR16PorDia",
    "classificacao/scripts",
):
    sys.path.insert(0, str(PROJECT_ROOT / sub))

from shapiro_normalidade import carregar  # noqa: E402
from reducao_colinearidade import spearman_matriz  # noqa: E402
from spearman_plsda_br16_manha import (  # noqa: E402
    escolher_componentes,
    escolher_representantes,
    vip_scores,
)
from classificacao_utils import PLSDAClassifier, selecionar_colunas  # noqa: E402

GENOTIPOS = ["BR16", "CD202", "EMB48"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]

ESTAGIO = "normalizado"
TURNO = "manha"
ALVO = "condicao"
CLASSE_POSITIVA = "NIRRIG"
ALPHA = 0.05
TOP_K = 5
SEMENTE = 42

COMPARACAO_ESTRESSE = (
    PROJECT_ROOT / "testeDiferencaSignificativa" / "dataset_gerado"
    / "comparacao_estresse.csv"
)

_ESTAT_CACHE: pd.DataFrame | None = None


def carregar_dados() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Meta, espectro normalizado e comprimentos de onda do turno da manha."""
    return carregar(ESTAGIO, turno=TURNO)


def carregar_estatisticas() -> pd.DataFrame:
    """p/q de IRRIG vs NIRRIG por dia x genotipo x banda."""
    global _ESTAT_CACHE
    if _ESTAT_CACHE is None:
        if not COMPARACAO_ESTRESSE.exists():
            raise SystemExit(f"Arquivo nao encontrado: {COMPARACAO_ESTRESSE}")
        _ESTAT_CACHE = pd.read_csv(COMPARACAO_ESTRESSE, sep=";")
    return _ESTAT_CACHE


def selecionar_top_dia(
    genotipo: str,
    dia: str,
    meta: pd.DataFrame,
    espectro: np.ndarray,
    w: np.ndarray,
    estat: pd.DataFrame,
    top_k: int = TOP_K,
) -> dict[str, object]:
    """Top-K bandas do genotipo num dia, pela regua do projeto.

    1. Correlacao de Spearman entre todas as bandas do genotipo no dia;
    2. bandas contiguas com |rho| > 0,80 dentro de janela de 10 nm formam
       grupos, e o representante de cada um e a banda de menor p (desempate
       pelo medoide);
    3. dentre as representantes com q_FDR <= 0,05, as `top_k` de maior VIP do
       PLS-DA (numero de componentes por leave-one-block-out).
    """
    mask = (meta["genotipo"] == genotipo) & (meta["dia"] == dia)
    meta_gd = meta.loc[mask].reset_index(drop=True)
    espectro_gd = espectro[mask]

    if len(meta_gd) < 8 or len(np.unique(meta_gd["condicao"])) < 2:
        return {"genotipo": genotipo, "dia": dia, "bandas": [], "top": None,
                "n_representantes": 0, "n_significativas": 0}

    estat_gd = (
        estat[(estat["genotipo"] == genotipo) & (estat["dia"] == dia)]
        .set_index("banda_nm").reindex(w.astype(int)).reset_index()
    )
    p = estat_gd["p_valor"].to_numpy()

    corr = spearman_matriz(espectro_gd)
    grupos, reps = escolher_representantes(corr, w, p)

    X = espectro_gd[:, reps]
    y = (meta_gd[ALVO].to_numpy() == CLASSE_POSITIVA).astype(int)
    n_comp, _ = escolher_componentes(X, y, meta_gd["bloco"].to_numpy())
    pls = PLSRegression(n_components=n_comp, scale=True).fit(X, y)

    resultado = estat_gd.loc[reps, [
        "banda_nm", "p_valor", "q_fdr", "epsilon2", "delta_cliff", "significativa"
    ]].copy()
    resultado["vip_pls_da"] = vip_scores(pls)

    significantes = resultado["significativa"].to_numpy()
    if significantes.any():
        top = resultado[significantes].nlargest(top_k, "vip_pls_da").copy()
        top["posicao"] = np.arange(1, len(top) + 1)
        bandas = top.sort_values("posicao")["banda_nm"].astype(int).tolist()
    else:
        top = pd.DataFrame()
        bandas = []

    return {
        "genotipo": genotipo,
        "dia": dia,
        "bandas": bandas,
        "top": top,
        "n_representantes": int(reps.sum()),
        "n_significativas": int(significantes.sum()),
        "n_componentes_pls_da": n_comp,
    }


def unir_bandas(selecoes: list[dict[str, object]]) -> list[int]:
    """Uniao das bandas dos genotipos de treino, preservando a ordem."""
    bandas: list[int] = []
    vistas: set[int] = set()
    for selecao in selecoes:
        for banda in selecao["bandas"]:
            if banda not in vistas:
                vistas.add(banda)
                bandas.append(banda)
    return bandas


def criar_modelos() -> dict[str, object]:
    """PLS-DA (n_comp=2) e Random Forest balanceado, com a regra do projeto."""
    return {
        "PLS-DA": PLSDAClassifier(n_components=2),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=1,
            random_state=SEMENTE,
        ),
    }


def obter_score(modelo, X: np.ndarray) -> np.ndarray:
    """Score da classe positiva para a curva ROC."""
    classes = list(modelo.classes_)
    pos = classes.index(CLASSE_POSITIVA)
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, pos]
    return modelo.decision_function(X)


def calcular_metricas(
    modelo: str,
    dia: str,
    fold: int,
    genotipo_treino: str,
    genotipo_teste: str,
    y_real: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    """Metricas de um fold, com balanced accuracy alem das padrao."""
    y_bin = (y_real == CLASSE_POSITIVA).astype(int)
    return {
        "modelo": modelo,
        "dia": dia,
        "fold": fold,
        "genotipo_treino": genotipo_treino,
        "genotipo_teste": genotipo_teste,
        "n_amostras_teste": len(y_real),
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, pos_label=CLASSE_POSITIVA,
                                      zero_division=0),
        "recall": recall_score(y_real, y_pred, pos_label=CLASSE_POSITIVA,
                               zero_division=0),
        "f1_score": f1_score(y_real, y_pred, pos_label=CLASSE_POSITIVA,
                             zero_division=0),
        "kappa": cohen_kappa_score(y_real, y_pred),
        "auc_roc": roc_auc_score(y_bin, score),
    }


def resumo_metricas(df: pd.DataFrame, agrupar_por: list[str]) -> pd.DataFrame:
    """Media e desvio das metricas numericas, agrupadas pelas colunas dadas."""
    numericas = [
        "n_amostras_teste", "accuracy", "balanced_accuracy", "precision",
        "recall", "f1_score", "kappa", "auc_roc",
    ]
    resumo = (
        df.groupby(agrupar_por, as_index=False)[numericas]
        .agg(["mean", "std"])
        .reset_index()
    )
    resumo.columns = [
        col if col in agrupar_por else f"{col}_{est}"
        for col, est in resumo.columns
    ]
    return resumo