#!/usr/bin/env python3
"""Curva espectral por dia para cada genótipo, na formatação da média temporal.

Mesma leitura de `plot_media_temporal_genotipo.py`: um painel por estágio
(recortado e normalizado), IRRIG e NIRRIG separados por cor, mesma tipografia e
legenda. A diferença é a unidade estatística do eixo:

- `plot_media_temporal_genotipo.py` colapsa os dias numa média temporal -- cada
  dia vira uma observação e o painel mostra a curva média do período.
- Este script mantém os dias abertos: cada dia de coleta vira uma curva própria,
  o tom de cor dentro da condição indo de claro (dia mais cedo) a escuro (dia
  mais tarde). Responde "como a curva evoluiu dia a dia", sem a média temporal.

Cada curva é a média espectral das leituras daquele genótipo, naquele dia, em
aquela condição (média da manhã). Uma figura por genótipo, colorida.

As figuras são salvas em graficosApresentacao/curvasPorDia/.

Uso:
    python plot_curvas_por_dia_genotipo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib não está instalado. Instale com: pip install matplotlib")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "testeDeNormalidade"))

from shapiro_normalidade import carregar_estagios  # noqa: E402
from plot_curvas_genotipo import ESTAGIOS  # noqa: E402
from plot_dispersao_dia_genotipo import CONDICOES, COR_CONDICAO  # noqa: E402

SAIDA_DIR = ROOT.parent / "graficosApresentacao" / "curvasPorDia"

TURNO = "manha"

# Rótulos com acentuação correta, já que a constante compartilhada não usa
# acentos ("Reflectancia"). A estrutura segue ESTAGIOS.
ROTULOS_ESTAGIO = {
    "recortado": ("Recortado (recorte + jump correction)", "Reflectância"),
    "normalizado": ("Normalizado (Savitzky-Golay + SNV)", "Reflectância (SNV)"),
}

# O dia mais cedo sai claro, o mais tarde escuro, dentro da cor da condição.
TOM_CLARO = 0.30  # fração da cor na mistura com branco no primeiro dia


def tom_do_dia(cor: str, t: float) -> tuple[float, float, float]:
    """Mistura a cor da condição com branco conforme o avanço dos dias."""
    r, g, b = mcolors.to_rgb(cor)
    return (t * r + (1 - t), t * g + (1 - t), t * b + (1 - t))


def plot_painel(
    ax: plt.Axes,
    w: np.ndarray,
    meta: pd.DataFrame,
    espectro: np.ndarray,
    genotipo: str,
    dias: list[str],
    titulo: str,
    unidade: str,
) -> None:
    """Um estágio: uma curva por dia, IRRIG e NIRRIG em tons da condição."""
    do_genotipo = (meta["genotipo"] == genotipo).to_numpy()
    n_dias = len(dias)
    for i, dia in enumerate(dias):
        t = 1.0 if n_dias == 1 else TOM_CLARO + (1 - TOM_CLARO) * i / (n_dias - 1)
        for condicao in CONDICOES:
            mask = do_genotipo & (meta["dia"] == dia).to_numpy()
            mask &= (meta["condicao"] == condicao).to_numpy()
            if mask.any():
                ax.plot(w, espectro[mask].mean(axis=0), color=tom_do_dia(COR_CONDICAO[condicao], t), linewidth=1.6)

    ax.set_title(titulo, fontsize=16, fontweight="bold")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=14)
    ax.set_ylabel(unidade, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)


def legenda_figura(fig: plt.Figure, dias: list[str]) -> None:
    """Legenda: cor = condição, tom = dia (claro para escuro)."""
    from matplotlib.lines import Line2D

    n_dias = len(dias)
    handles = [
        Line2D([], [], color=COR_CONDICAO[c], linewidth=1.6, label=c)
        for c in CONDICOES
    ]
    handles += [
        Line2D([], [], color=tom_do_dia("#555555", t), linewidth=1.6, label=dia)
        for dia, t in (
            (dia, 1.0 if n_dias == 1 else TOM_CLARO + (1 - TOM_CLARO) * i / (n_dias - 1))
            for i, dia in enumerate(dias)
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=14)


def gerar_figura(
    genotipo: str,
    meta: pd.DataFrame,
    estagios: dict[str, np.ndarray],
    w: np.ndarray,
) -> Path:
    """Gera e salva a figura de curvas por dia do genótipo."""
    do_genotipo = (meta["genotipo"] == genotipo).to_numpy()
    dias = sorted(meta.loc[do_genotipo, "dia"].unique().astype(str))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(ESTAGIOS), figsize=(7.5 * len(ESTAGIOS), 5.5))

    for ax, (chave, _, _) in zip(axes, ESTAGIOS):
        rotulo, unidade = ROTULOS_ESTAGIO[chave]
        plot_painel(ax, w, meta, estagios[chave], genotipo, dias, rotulo, unidade)

    legenda_figura(fig, dias)

    fig.suptitle(
        f"Genótipo {genotipo} - curva espectral por dia\n"
        f"Turno da manhã, IRRIG e NIRRIG separados  -  tom claro = dia mais "
        f"cedo, tom escuro = dia mais tarde  ({', '.join(dias)})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    saida = SAIDA_DIR / f"curvas_por_dia_{genotipo}.png"
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return saida


def main() -> None:
    print("Gerando figuras de curvas por dia...\n")
    meta, estagios, w = carregar_estagios(turno=TURNO)
    print(f"  {len(meta)} amostras do turno '{TURNO}', {len(w)} bandas "
          f"({w.min():.0f}-{w.max():.0f} nm)")

    dias = sorted(meta["dia"].unique())
    genotipos = sorted(meta["genotipo"].unique())
    print(f"  Dias: {', '.join(dias)}")
    print(f"  Genotipos: {', '.join(genotipos)}\n")

    for genotipo in genotipos:
        saida = gerar_figura(genotipo, meta, estagios, w)
        print(f"  {genotipo}: {saida.name}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
