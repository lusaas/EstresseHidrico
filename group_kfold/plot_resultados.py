#!/usr/bin/env python3
"""Figuras e tabela comparativa do cross-learning versus baseline intra.

Le as saidas de `cross_learning.py` e `baseline_intra.py`, monta
`outputs/comparativo.csv` (deficit de transferencia por dia x genotipo) e
gera em `outputs/figuras/`:

- `comparativo_por_dia_{DIA}.png`: balanced accuracy do cross-learning (barras
  por genotipo segurardo) frente ao teto intra-genotipo, para cada dia;
- `comparativo_todos_dias.png`: media dos 7 dias, com desvios;
- `transferencia_bandas.png`: heatmap da fracao do top-5 proprio do genotipo
  segurardo coberta pelas bandas selecionadas nos genotipos de treino.

Uso:
    python group_kfold/plot_resultados.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAIDA = ROOT / "outputs"
FIGURAS = SAIDA / "figuras"
CROSS = SAIDA / "cross_learning" / "metricas_folds.csv"
INTRA = SAIDA / "baseline_intra" / "metricas_folds.csv"

GENOTIPOS = ["BR16", "CD202", "EMB48"]
DIAS = ["D02", "D03", "D04", "D05", "D06", "D09", "D10"]
COR_CROSS = "#2ca02c"
COR_INTRA = "#1f77b4"


def carregar_folds(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise SystemExit(f"Arquivo nao encontrado: {caminho}")
    return pd.read_csv(caminho, sep=";")


def montar_comparativo() -> pd.DataFrame:
    cruzado = carregar_folds(CROSS)
    intra = carregar_folds(INTRA)

    cruzado = (
        cruzado.groupby(["dia", "genotipo_teste", "modelo"], as_index=False)
        .agg(cross_bacc=("balanced_accuracy", "mean"),
             cross_auc=("auc_roc", "mean"))
    )
    intra = (
        intra.groupby(["dia", "genotipo_teste", "modelo"], as_index=False)
        .agg(intra_bacc=("balanced_accuracy", "mean"),
             intra_auc=("auc_roc", "mean"))
    )

    comp = cruzado.merge(
        intra, on=["dia", "genotipo_teste", "modelo"], how="left"
    ).rename(columns={"genotipo_teste": "genotipo"})
    comp["deficit_bacc"] = comp["intra_bacc"] - comp["cross_bacc"]
    comp["deficit_auc"] = comp["intra_auc"] - comp["cross_auc"]

    saida = SAIDA / "comparativo.csv"
    comp.to_csv(saida, sep=";", index=False)
    print(f"Tabela comparativa salva em {saida}")
    return comp


def grafico_por_dia(comp: pd.DataFrame, dia: str) -> None:
    sub = comp[comp["dia"] == dia]
    y = np.arange(len(GENOTIPOS))
    altura = 0.32

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for i, genotipo in enumerate(GENOTIPOS):
        dados = sub[sub["genotipo"] == genotipo]
        if dados.empty:
            continue
        cross = dados["cross_bacc"].mean()
        intra = dados["intra_bacc"].mean()
        ax.barh(y[i] + altura / 2, cross, height=altura, color=COR_CROSS,
                label="Cross-learning" if i == 0 else None)
        ax.barh(y[i] - altura / 2, intra, height=altura, color=COR_INTRA,
                label="Intra-genótipo (teto)" if i == 0 else None)
        ax.text(cross + 0.015, y[i] + altura / 2, f"{cross:.2f}",
                va="center", fontsize=8, color=COR_CROSS)
        ax.text(intra + 0.015, y[i] - altura / 2, f"{intra:.2f}",
                va="center", fontsize=8, color=COR_INTRA)

    ax.set_yticks(y, GENOTIPOS)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Balanced accuracy (média dos modelos PLS-DA e Random Forest)")
    ax.set_title(f"Dia {dia} — cross-learning vs baseline intra-genótipo",
                 fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", color="#777777", alpha=0.20, linewidth=0.6)

    caminho = FIGURAS / f"comparativo_por_dia_{dia}.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em {caminho}")


def grafico_geral(comp: pd.DataFrame) -> None:
    media = (
        comp.groupby("genotipo", as_index=False)
        .agg(cross_bacc=("cross_bacc", "mean"),
             intra_bacc=("intra_bacc", "mean"),
             cross_desvio=("cross_bacc", "std"),
             intra_desvio=("intra_bacc", "std"))
    )
    y = np.arange(len(media))
    altura = 0.32

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.barh(y + altura / 2, media["cross_bacc"], height=altura, color=COR_CROSS,
            xerr=media["cross_desvio"], label="Cross-learning")
    ax.barh(y - altura / 2, media["intra_bacc"], height=altura, color=COR_INTRA,
            xerr=media["intra_desvio"], label="Intra-genótipo (teto)")
    for i, row in media.iterrows():
        ax.text(row["cross_bacc"] + row["cross_desvio"] + 0.015,
                y[i] + altura / 2, f"{row['cross_bacc']:.2f}", va="center",
                fontsize=8, color=COR_CROSS)
        ax.text(row["intra_bacc"] + row["intra_desvio"] + 0.015,
                y[i] - altura / 2, f"{row['intra_bacc']:.2f}", va="center",
                fontsize=8, color=COR_INTRA)

    ax.set_yticks(y, media["genotipo"])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Balanced accuracy (média sobre os 7 dias e os 2 modelos)")
    ax.set_title("Cross-learning vs baseline intra-genótipo — média dos dias",
                 fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", color="#777777", alpha=0.20, linewidth=0.6)

    caminho = FIGURAS / "comparativo_todos_dias.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em {caminho}")


def heatmap_transferencia() -> None:
    caminho = SAIDA / "cross_learning" / "transferencia_bandas.csv"
    if not caminho.exists():
        print("AVISO: transferencia_bandas.csv ausente; heatmap pulado.")
        return
    df = pd.read_csv(caminho, sep=";")
    tabela = df.pivot_table(index="dia", columns="genotipo_teste",
                            values="fracao_top5_proprio", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    dados = tabela.reindex(index=DIAS, columns=GENOTIPOS)
    im = ax.imshow(dados.to_numpy(dtype=float), cmap="YlGn", vmin=0, vmax=1,
                   aspect="auto")
    for i in range(len(DIAS)):
        for j in range(len(GENOTIPOS)):
            valor = dados.iloc[i, j]
            texto = f"{valor:.2f}" if np.isfinite(valor) else "—"
            ax.text(j, i, texto, ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(GENOTIPOS)), GENOTIPOS)
    ax.set_yticks(range(len(DIAS)), DIAS)
    ax.set_xlabel("Genótipo segurardo (teste)")
    ax.set_ylabel("Dia")
    ax.set_title("Fração do top-5 próprio do genótipo coberta pelas bandas "
                 "selecionadas nos genótipos de treino", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    caminho_saida = FIGURAS / "transferencia_bandas.png"
    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em {caminho_saida}")


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    comp = montar_comparativo()
    for dia in DIAS:
        grafico_por_dia(comp, dia)
    grafico_geral(comp)
    heatmap_transferencia()


if __name__ == "__main__":
    main()