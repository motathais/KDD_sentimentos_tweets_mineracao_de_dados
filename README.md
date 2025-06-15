-> Orientações gerais:

- Incluir a pasta "data" dentro de "projeto_mineracao"
- Subpastas de "data": external, processed, raw
- Incluir dados não tratados dentro de "raw"


-> Como criar venv:
1- Criar através do comando python -m venv venv
2- Ativar com .\venv\Scripts\activate

-> Estrutura das pastas:

projeto_mineracao/
├── data/
│   ├── raw/          # Dados originais (brutos) que você baixou, SEM ALTERAÇÃO
│   ├── processed/    # Dados tratados, limpos ou transformados
│   └── external/     # Caso use bases externas ou complementares no futuro
├── notebooks/        # Notebooks Jupyter para exploração e análises
├── src/              # Scripts Python (ex: limpeza, visualização, modelagem)
├── outputs/          # Gráficos, relatórios e arquivos gerados
└── README.md         # Explicação geral do projeto



