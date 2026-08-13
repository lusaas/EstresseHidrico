# Validação de cross-learning entre genótipos — Group K-Fold por genótipo

## Resumo

O conjunto de bandas mais significativas de estresse hídrico (IRRIG vs NIRRIG)
varia muito **entre genótipos** e **entre dias**. Este módulo pergunta uma coisa:
as bandas que separam o estresse em *alguns* genótipos conseguem separar o
estresse em *outro* genótipo que não participou da seleção?

Para responder, usamos **Group K-Fold com o grupo = genótipo** (leave-one-
genotype-out, 3 folds por dia): em cada fold, as bandas são selecionadas **somente
nos genótipos de treino**, um classificador é ajustado nesses genótipos e avaliado
no genótipo segurardo. O desempenho é comparado contra um **teto intra-genótipo**
(seleção e avaliação no mesmo material), que representa o melhor resultado sem
precisar de transferência.

Tudo — scripts, resultados e esta documentação — fica em `group_kfold/`.

---

## 1. Motivação

Nos dados do turno da manhã (`normalizado`), o contraste IRRIG vs NIRRIG é testado
banda a banda por dia × genótipo (`comparacao_estresse.csv`, Kruskal-Wallis + FDR).
A quantidade de bandas significativas (q ≤ 0,05) varia fortemente:

| genótipo | D02 | D03 | D04 | D05 | D06 | D09 | D10 |
|----------|-----|-----|-----|-----|-----|-----|-----|
| BR16     | 112 | 1876| 631 | 2026| 2030| 1100| 1289|
| CD202    | 2017| 1887| 1955| 1956| 1986| 1934| 1849|
| EMB48    | **0**| 1762| 1518| 1864| 1879| 1574| 1183|

E não é só a quantidade: a *localização espectral* das bandas difere entre materiais.
Isso motiva a pergunta do cross-learning: o "vocabulário espectral de estresse" de
um genótipo é compartilhado pelos demais?

### Por que o EMB48 não tem bandas significativas no D02?

O EMB48 no D02 é o único caso com **zero** bandas significativas. Não é problema de
dados: há 64 amostras válidas (32 IRRIG/32 NIRRIG, blocos B1–B4, nomenclatura
correta). É uma **resposta ao estresse fraca, difusa e retardada**:

- 403 bandas com p ≤ 0,05 brutos, mas a melhor (719 nm) tem p ≈ 7×10⁻⁴ e ε² = 0,17
  → após o FDR sobre as 2051 bandas, **q mínimo = 0,106 > 0,05**.
- O padrão de efeito do EMB48 D02 é **anti-correlacionado** ao do BR16 D02
  (correlação de ε² ≈ −0,54) e até ao do próprio D03 (−0,32): as janelas onde ele
  "quase separa" não são as janelas de estresse reais.
- A partir do D03 o EMB48 separa fortemente (1762–1879 bandas, ε² mediano
  0,26–0,45, correlação +0,80 com o BR16 D03): **onset tardio** da resposta.

No cross-learning, o EMB48 D02 entra como *teste puro*: o modelo treinado nas
bandas de BR16+CD202 é avaliado num dia em que o próprio EMB48 não apresenta a
assinatura espectral. Espera-se resultado perto do acaso, e o teto intra-genótipo
não existe para esse caso (o script o registra como 0,5 de balanced accuracy).

---

## 2. Conceito

**Cross-learning** (transferência entre genótipos): treinar um discriminante de
condição (IRRIG vs NIRRIG) com as bandas selecionadas nos genótipos de treino e
aplicá-lo nos dados do genótipo segurardo, usando as **mesmas bandas**.

- Se o desempenho no segurardo ≈ chance → as bandas **não transferem** (resposta
  genótipo-específica).
- Se o desempenho se aproxima do teto intra-genótipo → as bandas **transferem**
  (assinatura espectral de estresse compartilhada).

---

## 3. Metodologia passo a passo

### Etapa 1 — Dados
`carregar("normalizado", turno="manha")` carrega meta (nomenclaura, bloco,
genotipo, condicao, dia), espectro SNV e comprimentos de onda (2051 bandas,
400–2450 nm). A análise é **dia a dia**, nos 7 dias: D02–D06, D09, D10.

### Etapa 2 — Seleção de variáveis (in-fold)
Para cada **genótipo × dia**, as top-5 bandas são escolhidas com a régua do projeto
(igual a `analiseBR16PorDia/spearman_plsda_br16_manha.py`):

1. Matriz de correlação de Spearman entre todas as bandas do genótipo no dia;
2. bandas contíguas com |ρ| > 0,80 dentro de janela de 10 nm formam grupos; o
   representante de cada grupo é a banda de menor p (desempate pelo medoide);
3. entre as representantes com q_FDR ≤ 0,05, ficam as `TOP_K = 5` de maior **VIP do
   PLS-DA** (número de componentes escolhido por leave-one-block-out).

As features de cada fold são a **união dos top-5 dos genótipos de treino** (≤ 10
bandas). **A seleção nunca usa o genótipo segurardo.**

### Etapa 3 — Divisão por genótipo (Group K-Fold)
Em cada dia: `GroupKFold(n_splits=3)` com `groups = genotipo`. Cada fold segura um
genótipo inteiro fora do treino. Com 3 genótipos, isso é equivalente a
leave-one-genotype-out. O split é determinístico (sem shuffle).

### Etapa 4 — Modelos
Dois classificadores treinados nas amostras dos genótipos de treino:
- **PLS-DA** (`PLSDAClassifier`, n_components = 2);
- **Random Forest** (500 árvores, `class_weight="balanced"`, `random_state=42`).

### Etapa 5 — Avaliação
No genótipo segurardo, por fold: `accuracy, balanced_accuracy, precision, recall,
f1, kappa, auc_roc`. Resumo por dia × genótipo segurardo × modelo (média e desvio).

### Etapa 6 — Transferência de bandas
Para cada fold, registra a interseção das bandas selecionadas com:
- o **top-5 próprio** do genótipo segurardo (mesma régua, no seu próprio dia);
- o conjunto de **bandas significativas próprias** (q ≤ 0,05).

A fração `fracao_top5_proprio` diz quanto do "vocabulário" do segurardo foi
coberto pelas bandas dos outros. Quando o segurardo não tem top-5 próprio
(EMB48/D02), a fração é `NaN`.

### Etapa 7 — Teto intra-genótipo (baseline)
Para cada dia × genótipo, as top-5 do **próprio** genótipo são avaliadas com
`GroupKFold(n_splits=4)` por **bloco** (leave-one-block-out), mesmo par de modelos
e mesmas métricas. Representa o melhor sem transferência. Quando o genótipo não tem
bandas (EMB48/D02), o cenário é registrado com balanced accuracy = AUC = 0,5 e a
flag em `casos_sem_bandas.csv`.

### Etapa 8 — Comparativo
`deficit_bacc = intra_bacc − cross_bacc` (e análogo para AUC), por
dia × genótipo × modelo. Déficit > 0 → a transferência perde em relação a usar as
bandas do próprio material; próximo de 0 → transferência "gratuita".

---

## 4. Como executar

Na raiz do projeto (dependências: `pandas`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`):

```bash
# 1. Cross-learning (GroupKFold por genótipo, por dia)
python group_kfold/cross_learning.py

# 2. Teto intra-genótipo (GroupKFold por bloco)
python group_kfold/baseline_intra.py

# 3. Tabela comparativa + figuras
python group_kfold/plot_resultados.py
```

A ordem é obrigatória: o passo 3 lê as saídas dos passos 1 e 2.

---

## 5. Estrutura de resultados

```
group_kfold/
├── selecao_cross.py        # helpers: seleção top-5, modelos, métricas
├── cross_learning.py       # GroupKFold por genótipo (por dia)
├── baseline_intra.py       # teto intra-genótipo (por bloco)
├── plot_resultados.py      # comparativo + figuras
├── README.md               # este documento
└── outputs/
    ├── comparativo.csv     # déficit (intra − cross) por dia×genótipo×modelo
    ├── cross_learning/
    │   ├── metricas_folds.csv        # métricas por dia×fold×modelo (42 linhas)
    │   ├── metricas_resumo.csv       # média/desvio por dia×teste×modelo
    │   ├── transferencia_bandas.csv  # overlap das bandas selecionadas vs próprias
    │   ├── predicoes.csv             # predição por amostra de teste
    │   ├── fold_assignments.csv      # qual amostra ficou em qual fold
    │   └── configuracao.csv          # parâmetros da execução
    ├── baseline_intra/
    │   ├── metricas_folds.csv        # por dia×genótipo×fold×modelo
    │   ├── metricas_resumo.csv
    │   ├── predicoes.csv
    │   ├── casos_sem_bandas.csv      # EMB48/D02 e outros sem top-5
    │   └── configuracao.csv
    └── figuras/
        ├── comparativo_por_dia_{DIA}.png   # cross vs intra por dia (7)
        ├── comparativo_todos_dias.png      # média dos 7 dias
        └── transferencia_bandas.png        # heatmap da fração coberta
```

### Glossário das colunas principais
- `genotipo_treino` — genótipos que participaram da seleção/treino (ex.: `BR16+CD202`).
- `genotipo_teste` — genótipo segurardo (avaliado com as bandas dos outros).
- `balanced_accuracy` — média da sensibilidade das duas classes; é a métrica principal
  (a precisão bruta é enganosa com classes desbalanceadas).
- `fracao_top5_proprio` — proporção do top-5 do próprio segurardo coberta pelas bandas
  selecionadas no treino; `NaN` quando o segurardo não tem top-5 (EMB48/D02).
- `deficit_bacc` — `intra − cross`; > 0 = perda por transferir, ≈ 0 = transferência boa.

---

## 6. Leitura dos resultados

Exemplo real (balanced accuracy média sobre os 2 modelos; **cross** e **intra**):

| dia | BR16 cross/intra | CD202 cross/intra | EMB48 cross/intra |
|-----|------------------|-------------------|-------------------|
| D02 | 0,59 / 0,77      | 0,73 / 0,63       | 0,56 / 0,50*      |
| D03 | 0,69 / 0,80      | 0,89 / 0,91       | 0,89 / 0,95       |
| D04 | 0,80 / 0,75      | 0,83 / 0,77       | 0,91 / 0,81       |
| D05 | 0,88 / 1,00      | 0,83 / 1,00       | 0,84 / 0,75       |
| D06 | 1,00 / 1,00      | 0,95 / 0,98       | 0,95 / 0,83       |
| D09 | 0,52 / 0,89      | 0,89 / 1,00       | 0,86 / 0,76       |
| D10 | 0,45 / 0,56      | 0,74 / 0,86       | 0,72 / 0,97       |

\* EMB48/D02 não tem top-5 próprio; o "intra" é o valor sentinela 0,5.

Leituras esperadas:
- **CD202 é o genótipo que mais transfere para os outros** (déficit médio pequeno,
  0,04): seu top-5 é o melhor representante do "vocabulário de estresse" comum.
- **BR16 é o que mais perde** quando avaliado com bandas alheias (déficit 0,12),
  sobretudo em D09/D10: sua resposta é mais idiossincrática nesses dias.
- **EMB48/D02 perto do acaso (0,56)** confirma que o dia não tem assinatura de
  estresse — nem própria nem transferível.
- A fração do top-5 próprio coberta pelas bandas dos treino costuma ser baixa
  (heatmap): as bandas transferem *funcionalmente* (separam a condição) mesmo sem
  coincidir com o top-5 do segurardo.

---

## 7. Limitações

- **Somente 3 genótipos** → k = 3 folds, treino sempre em 2 genótipos; não há como
  generalizar para materiais não observados além dessa janela.
- **Pseudorreplicação**: blocos de campo se repetem entre dias; como o split é por
  genótipo inteiro, não há vazamento treino/teste por essa via, mas o desempenho
  pode ser otimista quanto à replicabilidade experimental (n efetivo ≈ 4 blocos).
- **Top-5 por VIP** é um recorte rígido; outras métricas de seleção (todas as
  significativas, Boruta) dariam outro cenário.
- **EMB48/D02** não tem bandas significativas; o baseline intra desse caso é um
  sentinela 0,5, não um desempenho real.
- Métricas de classificação usam amostras (não médias por bloco) como unidade, como
  no restante do projeto; os tamanhos de efeito devem ser usados para dimensionar a
  magnitude além do p.

---

## 8. Reprodução

- Estágio `normalizado`, turno `manha`, seed `42` (Random Forest).
- Seleção idêntica à de `analiseBR16PorDia/spearman_plsda_br16_manha.py`.
- `GroupKFold` é determinístico (sem shuffle).
- Dependências: `pandas>=3`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.
