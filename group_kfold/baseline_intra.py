#!/usr/bin/env python3
"""Baseline intra-genotipo: o teto sem transferencia entre materiais.

Para cada dia x genotipo, classifica IRRIG vs NIRRIG usando as top-K bandas
*do proprio genotipo* naquele dia, com validacao leave-one-block-out
(GroupKFold com grupo = bloco, 4 folds). Isso representa o melhor desempenho
possivel sem precisar transferir conhecimento entre genotipos -- e a
referencia contra a qual o cross-learning de `cross_learning.py` deve ser
comparado. O deficit (intra - cross) mede o quanto a transferencia perde.

Caso o genotipo nao tenha banda significativa no dia (ex.: EMB48 no D02), o
cenario e registrado com balanced accuracy = 0,5 e auc = 0,5 e a flag
`sem_bandas=1`, sem treinar modelo.

Saidas em `outputs/baseline_intra/`: metricas_folds.csv, metricas_resumo.csv,
predicoes.csv (vazia nos casos sem bandas), configuracao.csv.

Uso:
    python group_kfold/baseline_intra.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from selecao_cross import (
    DIAS,
    GENOTIPOS,
    ALVO,
    CLASSE_POSITIVA,
    carregar_dados,
    carregar_estatisticas,
    calcular_metricas,
    criar_modelos,
    obter_score,
    resumo_metricas,
    selecionar_colunas,
    selecionar_top_dia,
)

ROOT = Path(__file__).resolve().parent
SAIDA = ROOT / "outputs" / "baseline_intra"

N_BLOCOS = 4


def main() -> None:
    meta, espectro, w = carregar_dados()
    estat = carregar_estatisticas()
    modelos_base = criar_modelos()
    cache: dict[tuple[str, str], dict[str, object]] = {}

    SAIDA.mkdir(parents=True, exist_ok=True)

    metricas, predicoes, sem_bandas = [], [], []

    print(f"Dados: {len(meta)} amostras; baseline intra-genotipo "
          f"(GroupKFold por bloco, {N_BLOCOS} folds)")

    for dia in DIAS:
        for genotipo in GENOTIPOS:
            chave = (genotipo, dia)
            if chave not in cache:
                cache[chave] = selecionar_top_dia(
                    genotipo, dia, meta, espectro, w, estat
                )
            selecao = cache[chave]
            bandas = selecao["bandas"]

            mask = (meta["genotipo"] == genotipo) & (meta["dia"] == dia)
            meta_gd = meta.loc[mask].reset_index(drop=True)
            X = espectro[mask]
            y = meta_gd[ALVO].astype(str).to_numpy()
            grupos = meta_gd["bloco"].astype(str).to_numpy()

            if not bandas or len(np.unique(grupos)) < 2:
                for nome in modelos_base:
                    y_bin = (y == CLASSE_POSITIVA).astype(int)
                    metricas.append({
                        "modelo": nome,
                        "dia": dia,
                        "fold": 0,
                        "genotipo_treino": genotipo,
                        "genotipo_teste": genotipo,
                        "n_amostras_teste": len(y),
                        "accuracy": y_bin.mean(),
                        "balanced_accuracy": 0.5,
                        "precision": y_bin.mean(),
                        "recall": 1.0,
                        "f1_score": 2 * y_bin.mean() / (1 + y_bin.mean())
                                    if y_bin.mean() > 0 else 0.0,
                        "kappa": 0.0,
                        "auc_roc": 0.5,
                    })
                sem_bandas.append({
                    "dia": dia, "genotipo": genotipo, "n_bandas": len(bandas),
                    "motivo": "sem bandas significativas" if not bandas
                              else "menos de 2 blocos",
                })
                print(f"  {dia} {genotipo}: sem top-5 (n_bandas={len(bandas)})"
                      f" -- baseline = acaso")
                continue

            X_b = selecionar_colunas(X, w, bandas)
            cv = GroupKFold(n_splits=min(N_BLOCOS, len(np.unique(grupos))))
            for fold, (tr_idx, te_idx) in enumerate(cv.split(X_b, y, groups=grupos), start=1):
                for nome, modelo_base in modelos_base.items():
                    modelo = clone(modelo_base)
                    modelo.fit(X_b[tr_idx], y[tr_idx])
                    y_pred = modelo.predict(X_b[te_idx])
                    score = obter_score(modelo, X_b[te_idx])

                    metricas.append(calcular_metricas(
                        nome, dia, fold, genotipo, genotipo,
                        y[te_idx], y_pred, score,
                    ))

                    df_pred = meta_gd.loc[te_idx].copy()
                    df_pred.insert(0, "indice_amostra",
                                   meta.index.to_numpy()[mask][te_idx])
                    df_pred.insert(0, "modelo", nome)
                    df_pred.insert(0, "fold", fold)
                    df_pred["classe_real"] = y[te_idx]
                    df_pred["classe_predita"] = y_pred
                    df_pred[f"score_{CLASSE_POSITIVA}"] = score
                    df_pred["acerto"] = y_pred == y[te_idx]
                    for pos, banda in enumerate(bandas):
                        df_pred[str(banda)] = X_b[te_idx, pos]
                    predicoes.append(df_pred)

            print(f"  {dia} {genotipo}: {len(bandas)} bandas, "
                  f"n_rep={selecao['n_representantes']}, "
                  f"n_sig={selecao['n_significativas']}")

    df_metricas = pd.DataFrame(metricas)
    df_resumo = resumo_metricas(df_metricas, ["dia", "genotipo_teste", "modelo"])

    df_metricas.to_csv(SAIDA / "metricas_folds.csv", sep=";", index=False)
    df_resumo.to_csv(SAIDA / "metricas_resumo.csv", sep=";", index=False)
    if predicoes:
        pd.concat(predicoes, ignore_index=True).to_csv(
            SAIDA / "predicoes.csv", sep=";", index=False
        )
    pd.DataFrame(sem_bandas).to_csv(SAIDA / "casos_sem_bandas.csv", sep=";", index=False)
    pd.DataFrame([{
        "estrategia_validacao": "GroupKFold (grupo = bloco), leave-one-block-out",
        "k_folds": N_BLOCOS,
        "selecao": "top5 VIP do proprio genotipo no dia (q_FDR<=0,05)",
        "alvo": ALVO,
        "classe_positiva": CLASSE_POSITIVA,
        "modelos": ", ".join(modelos_base),
        "turno": "manha",
        "estagio": "normalizado",
        "semente": 42,
    }]).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)

    print("\nResumo intra-genotipo (balanced accuracy media por dia x genotipo):")
    pivot = (
        df_resumo.pivot_table(
            index="dia", columns="genotipo_teste",
            values="balanced_accuracy_mean", aggfunc="mean",
        )
    )
    print(pivot.round(3).to_string())
    print(f"\nResultados salvos em {SAIDA}")


if __name__ == "__main__":
    main()