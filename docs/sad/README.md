# docs/sad — documentacao especifica de SAD

## Status atual — CONCLUIDO

[FEATURE_SELECTION.md](FEATURE_SELECTION.md) — **selecao de features e tuning
de hiperparametros concluidos**. Conclusao: nenhuma feature removida (18
remocoes individuais testadas, todas pioraram); `max_depth=4` (arvores mais
rasas) foi o unico ajuste que ajudou — o oposto de GAD, que precisou de
arvores mais profundas. Configuracao final: 17 features, max_depth=4,
Kappa 0.8415 (era 0.7788 sem tuning).

Ainda faltam (seguindo o padrao de `docs/gad/`):

- `manifesto_resultados_oficiais.md` (quando houver resultados oficiais de SAD)
- `dissertacao/` e `slides/` especificos de SAD, se a secao entrar na
  dissertacao ou em apresentacoes futuras

Documentos que ja cobrem GAD e SAD juntos (`docs/melhores_resultados.txt`,
`docs/relatorio_preprocessamento_resultados.md`, `docs/dataset/`,
`docs/slides/apresentacao_1/`) continuam em `docs/` (raiz), pois nao fazem
sentido duplicados.
