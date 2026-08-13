#!/usr/bin/env python3
"""Cross-learning: Group K-Fold por genotipo, dia a dia.

Para cada dia de coleta, as amostras sao divididas pelo genotipo
(GroupKFold com grupo = genotipo, 3 folds -- cada fold segura um material
inteiro fora do treino). Em cada fold:

1. Selecao in-fold: para cada genotipo de treino, as top-K banda do dia sao
   escolhidas com a regua do projeto (representantes Spearman, q_FDR <= 0,05
   e maior VIP do PLS-DA). As features do fold sao a uniao desses top-K.
   O genotipo segurardo nunca participa da selecao.
2. Treino: PLS-DA e Random Forest nas amostras dos genotipos de treino,
   usando as bandas selecionadas (contraste IRRIG vs NIRRIG).
3. Teste: as mesmas bandas sao avaliadas nas amostras do genotipo segurardo.
4. Transferencia: registra quantas bandas selecionadas estao no top-5 proprio
   do genotipo segurardo e nas suas bandas significativas do dia -- o quanto
   o vocabulario espectrais de stress transferiu.

Saidas em `outputs/cross_learning/`: metricas_folds.csv, metricas_resumo.csv,
transferencia_bandas.csv, predicoes.csv, fold_assignments.csv, configuracao.csv.

Uso:
    python group_kfold/cross_learning.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

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
    unir_bandas,
)

ROOT = Path(__file__).resolve().parent
SAIDA = ROOT / "outputs" / "cross_learning"

N_FOLDS = 3


def main() -> None:
    meta, espectro, w = carregar_dados()
    estat = carregar_estatisticas()
    modelos_base = criar_modelos()
    cache: dict[tuple[str, str], dict[str, object]] = {}

    SAIDA.mkdir(parents=True, exist_ok=True)

    metricas, predicoes, atribuicoes, transferencias = [], [], [], []

    print(f"Dados: {len(meta)} amostras do turno da manha; genotipos: "
          f"{', '.join(GENOTIPOS)}; folds por dia: {N_FOLDS} (grupo = genotipo)")

    for dia in DIAS:
        mask = meta["dia"].eq(dia).to_numpy()
        meta_d = meta.loc[mask].reset_index(drop=True)
        X_d = espectro[mask]
        y_d = meta_d[ALVO].astype(str).to_numpy()
        grupos_d = meta_d["genotipo"].astype(str).to_numpy()

        cv = GroupKFold(n_splits=N_FOLDS)
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X_d, y_d, groups=grupos_d), start=1):
            gen_train = sorted(set(grupos_d[tr_idx]))
            gen_test = sorted(set(grupos_d[te_idx]))
            if len(gen_test) != 1:
                raise RuntimeError(f"{dia} fold {fold}: esperado 1 genotipo de teste")

            selecoes = []
            for g in gen_train:
                chave = (g, dia)
                if chave not in cache:
                    cache[chave] = selecionar_top_dia(
                        g, dia, meta, espectro, w, estat
                    )
                selecoes.append(cache[chave])
            bandas = unir_bandas(selecoes)

            if not bandas:
                print(f"  {dia} fold {fold}: sem bandas selecionadas em "
                      f"{'+'.join(gen_train)} -- fold descartado")
                continue

            X_tr = selecionar_colunas(X_d[tr_idx], w, bandas)
            X_te = selecionar_colunas(X_d[te_idx], w, bandas)
            y_tr, y_te = y_d[tr_idx], y_d[te_idx]

            meta_teste = meta_d.loc[te_idx].copy()
            meta_teste.insert(0, "indice_amostra", meta.index.to_numpy()[mask][te_idx])
            metic_treino = "+".join(gen_train)

            for split, indices in [("treino", tr_idx), ("teste", te_idx)]:
                df_aux = meta_d.loc[indices].copy()
                df_aux.insert(0, "indice_amostra", meta.index.to_numpy()[mask][indices])
                df_aux.insert(0, "split", split)
                df_aux.insert(0, "genotipo_teste", gen_test[0])
                df_aux.insert(0, "genotipo_treino", metic_treino)
                df_aux.insert(0, "fold", fold)
                atribuicoes.append(df_aux)

            for nome, modelo_base in modelos_base.items():
                modelo = clone(modelo_base)
                modelo.fit(X_tr, y_tr)
                y_pred = modelo.predict(X_te)
                score = obter_score(modelo, X_te)

                metricas.append(calcular_metricas(
                    nome, dia, fold, metic_treino, gen_test[0],
                    y_te, y_pred, score,
                ))

                df_pred = meta_teste.copy()
                df_pred.insert(0, "modelo", nome)
                df_pred.insert(0, "genotipo_teste", gen_test[0])
                df_pred.insert(0, "genotipo_treino", metic_treino)
                df_pred.insert(0, "fold", fold)
                df_pred["classe_real"] = y_te
                df_pred["classe_predita"] = y_pred
                df_pred[f"score_{CLASSE_POSITIVA}"] = score
                df_pred["acerto"] = y_pred == y_te
                for pos, banda in enumerate(bandas):
                    df_pred[str(banda)] = X_te[:, pos]
                predicoes.append(df_pred)

            # Transferencia de bandas contra o proprio genotipo segurardo.
            propio = cache.get(
                (gen_test[0], dia),
                selecionar_top_dia(gen_test[0], dia, meta, espectro, w, estat),
            )
            top_proprio = set(propio["bandas"])
            sig_proprias = set(
                estat[(estat.genotipo == gen_test[0]) & (estat.dia == dia)
                      & (estat.q_fdr <= 0.05)]["banda_nm"].astype(int)
            )
            selecionadas = set(bandas)
            transferencias.append({
                "dia": dia,
                "fold": fold,
                "genotipos_treino": metic_treino,
                "genotipo_teste": gen_test[0],
                "bandas_selecionadas": ";".join(str(b) for b in bandas),
                "n_bandas": len(bandas),
                "n_top5_proprio": len(top_proprio),
                "n_sobrepostas_top5_proprio": len(selecionadas & top_proprio),
                "fracao_top5_proprio": (
                    len(selecionadas & top_proprio) / len(top_proprio)
                    if top_proprio else float("nan")
                ),
                "n_significativas_proprias": len(sig_proprias),
                "n_sobrepostas_significativas": len(selecionadas & sig_proprias),
            })

            print(f"  {dia} fold {fold}: teste={gen_test[0]} | treino="
                  f"{'+'.join(gen_train)} | {len(bandas)} bandas")

    df_metricas = pd.DataFrame(metricas)
    df_resumo = resumo_metricas(df_metricas, ["dia", "genotipo_teste", "modelo"])

    df_metricas.to_csv(SAIDA / "metricas_folds.csv", sep=";", index=False)
    df_resumo.to_csv(SAIDA / "metricas_resumo.csv", sep=";", index=False)
    pd.concat(predicoes, ignore_index=True).to_csv(
        SAIDA / "predicoes.csv", sep=";", index=False
    )
    pd.concat(atribuicoes, ignore_index=True).to_csv(
        SAIDA / "fold_assignments.csv", sep=";", index=False
    )
    pd.DataFrame(transferencias).to_csv(
        SAIDA / "transferencia_bandas.csv", sep=";", index=False
    )
    pd.DataFrame([{
        "estrategia_validacao": "GroupKFold (grupo = genotipo), leave-one-genotype-out",
        "k_folds_por_dia": N_FOLDS,
        "selecao": "representantes Spearman + q_FDR<=0,05 + top5 VIP (PLS-DA)",
        "alvo": ALVO,
        "classe_positiva": CLASSE_POSITIVA,
        "modelos": ", ".join(modelos_base),
        "turno": "manha",
        "estagio": "normalizado",
        "semente": 42,
        "total_folds": len(df_metricas),
    }]).to_csv(SAIDA / "configuracao.csv", sep=";", index=False)

    print("\nResumo cross-learning (balanced accuracy media por dia x genotipo teste):")
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