#!/usr/bin/env python3
"""Curva por dia com IC de 95%, um PNG por genótipo.

Versão dia a dia de `plot_media_temporal_genotipo.py`: em vez de colapsar os
dias numa média temporal, cada dia de coleta ganha um subplot com a mesma
leitura daquele script -- linha grossa = média espectral, faixa hachurada = IC
de 95% da média, linha pontilhada fina = mínimo e máximo observados -- para
IRRIG e NIRRIG separados por cor.

Grid com uma linha por dia (D02 a D10) e duas colunas (recortado e normalizado).
A escala do eixo y é compartilhada por estágio: sem isso cada dia escolhe o
próprio limite e a comparação entre dias some. Uma figura por genótipo.

As figuras são salvas em graficosApresentacao/curvasPorDiaComIC/.

Uso:
    python plot_curvas_por_dia_ic_genotipo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib não está instalado. Instale com: pip install matplotlib")

from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402
from plot_curvas_genotipo import ESTAGIOS  # noqa: E402
from plot_dispersao_dia_genotipo import CONDICOES, COR_CONDICAO  # noqa: E402

SAIDA_DIR = ROOT.parent / "graficosApresentacao" / "curvasPorDiaComIC"

TURNO = "manha"
CONFIANCA = 0.95

# Rótulos com acentuação correta, já que a constante compartilhada não usa
# acentos ("Reflectancia"). A estrutura segue ESTAGIOS.
ROTULOS_ESTAGIO = {
    "recortado": ("Recortado (recorte + jump correction)", "Reflectância"),
    "normalizado": ("Normalizado (Savitzky-Golay + SNV)", "Reflectância (SNV)"),
}

# Hachuras opostas: onde as duas faixas se cruzam, mesma direção viraria uma
# textura só e a sobreposição ficaria ilegível.
HACHURA = {"IRRIG": "///", "NIRRIG": "\\\\\\"}


def resumo_banda(espectro: np.ndarray) -> dict[str, np.ndarray]:
    """Média, limites do IC de 95% da média, mínimo e máximo por banda."""
    n = espectro.shape[0]
    media = espectro.mean(axis=0)
    minimo = espectro.min(axis=0)
    maximo = espectro.max(axis=0)

    if n > 1:
        erro = espectro.std(axis=0, ddof=1) / np.sqrt(n)
        margem = t_dist.ppf(0.5 + CONFIANCA / 2, df=n - 1) * erro
    else:
        margem = np.zeros_like(media)

    return {
        "media": media,
        "ic_inf": media - margem,
        "ic_sup": media + margem,
        "min": minimo,
        "max": maximo,
        "n": n,
    }


def plot_celula(
    ax: plt.Axes,
    w: np.ndarray,
    resumo: dict[str, dict],
    titulo: str,
    unidade: str,
) -> None:
    """Um dia: média, IC hachurado e envelope min/max das duas condições."""
    for condicao in CONDICOES:
        if condicao not in resumo:
            continue
        cor = COR_CONDICAO[condicao]
        r = resumo[condicao]
        ax.fill_between(w, r["ic_inf"], r["ic_sup"], facecolor="none",
                        edgecolor=cor, hatch=HACHURA[condicao], linewidth=0.0)
        ax.plot(w, r["ic_inf"], color=cor, linewidth=0.8)
        ax.plot(w, r["ic_sup"], color=cor, linewidth=0.8)
        ax.plot(w, r["min"], color=cor, linewidth=0.7, linestyle=(0, (1, 2)))
        ax.plot(w, r["max"], color=cor, linewidth=0.7, linestyle=(0, (1, 2)))

    for condicao in CONDICOES:
        if condicao not in resumo:
            continue
        ax.plot(w, resumo[condicao]["media"], color=COR_CONDICAO[condicao],
                linewidth=2.2, solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white"),
                              pe.Normal()])

    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=8)
    ax.set_ylabel(unidade, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def legenda_figura(fig: plt.Figure) -> None:
    """Cor = condição; traço/hachura = estatística."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    neutro = "#555555"
    handles = [
        Line2D([], [], color=COR_CONDICAO[c], linewidth=2.2, label=c)
        for c in CONDICOES
    ]
    handles += [
        Line2D([], [], color=neutro, linewidth=2.2, label="Média"),
        Patch(facecolor="none", edgecolor=neutro, hatch="///",
              label="IC 95% da média"),
        Line2D([], [], color=neutro, linewidth=0.7, linestyle=(0, (1, 2)),
               label="Mínimo / máximo"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=11)


def gerar_figura(
    genotipo: str,
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
    dias: list[str],
) -> Path:
    """Gera e salva a figura por dia com IC de um genótipo."""
    do_genotipo = (meta["genotipo"] == genotipo).to_numpy()

    resumos: dict[tuple[str, str], dict[str, dict]] = {}
    for chave, _, _ in ESTAGIOS:
        for dia in dias:
            do_dia = do_genotipo & (meta["dia"] == dia).to_numpy()
            por_condicao = {}
            for condicao in CONDICOES:
                mask = do_dia & (meta["condicao"] == condicao).to_numpy()
                if mask.any():
                    por_condicao[condicao] = resumo_banda(estagios[chave][mask])
            if por_condicao:
                resumos[(chave, dia)] = por_condicao

    # Escala de y compartilhada por estágio: sem isso cada dia escolhe o
    # próprio limite e a comparação entre dias -- o ponto do grid -- some.
    limites = {}
    for chave, _, _ in ESTAGIOS:
        valores = np.concatenate(
            [r[k] for (c, _), por_condicao in resumos.items() if c == chave
             for r in por_condicao.values() for k in ("min", "max")]
        )
        folga = 0.05 * (valores.max() - valores.min())
        limites[chave] = (valores.min() - folga, valores.max() + folga)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(dias), len(ESTAGIOS),
                             figsize=(7.5 * len(ESTAGIOS), 3.4 * len(dias)))

    for i, dia in enumerate(dias):
        for j, (chave, _, _) in enumerate(ESTAGIOS):
            ax = axes[i, j]
            resumo = resumos.get((chave, dia))
            if resumo is None:
                ax.set_axis_off()
                continue
            rotulo, unidade = ROTULOS_ESTAGIO[chave]
            ns = ", ".join(f"{c} n={resumo[c]['n']}" for c in CONDICOES
                           if c in resumo)
            plot_celula(ax, w, resumo, f"{dia} - {rotulo}  ({ns})", unidade)
            ax.set_ylim(*limites[chave])

    legenda_figura(fig)

    fig.suptitle(
        f"Genótipo {genotipo} - curva espectral por dia com IC de 95%\n"
        "Turno da manhã, IRRIG e NIRRIG separados  -  escala do eixo y "
        "compartilhada por estágio",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.025, 1, 0.975])
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    saida = SAIDA_DIR / f"curvas_por_dia_ic_{genotipo}.png"
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return saida


def main() -> None:
    print("Gerando figuras por dia com IC de 95%...\n")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}', {len(w)} bandas "
          f"({w.min():.0f}-{w.max():.0f} nm)")

    dias = sorted(meta["dia"].unique())
    genotipos = sorted(meta["genotipo"].unique())
    print(f"  Dias: {', '.join(dias)}")
    print(f"  Genotipos: {', '.join(genotipos)}\n")

    for genotipo in genotipos:
        saida = gerar_figura(genotipo, meta, estagios, w, dias)
        print(f"  {genotipo}: {saida.name}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
