# Projeto de Mineração de Dados

## Orientações Gerais

- Incluir a pasta `data` dentro de `projeto_mineracao`.
- Subpastas de `data`:
  - `external`: Dados externos ou complementares.
  - `processed`: Dados tratados ou transformados.
  - `raw`: Dados originais, **não tratados**.
- Incluir dados **não tratados** dentro da pasta `raw`.

---
## 📂 Estrutura de Pastas

```text
projeto_mineracao/
├── data/
│   ├── raw/          # Dados originais (brutos), sem alteração
│   ├── processed/    # Dados tratados, limpos ou transformados
│   └── external/     # Dados externos ou complementares
├── notebooks/        # Notebooks Jupyter para exploração e análises
├── src/              # Scripts Python (ex: limpeza, visualização, modelagem)
├── outputs/          # Gráficos, relatórios e arquivos gerados
└── README.md         # Explicação geral do projeto

--
## Como Criar Ambiente Virtual Python

1. Criar o ambiente virtual:
   ```bash
   python -m venv venv

2. Ativar ambiente virtual: .\venv\Scripts\activate



