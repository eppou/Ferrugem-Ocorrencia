# Model Bundle

## Executar o projeto
1. Instalar o [Poetry](https://python-poetry.org)
2. Executar `poetry install` na raiz do projeto
3. Executar `poetry shell` para entrar no ambiente virtual configurado no Poetry 
4. Rodar o projeto com `python main.py <parametro>`, onde `<parametro>` é alguma das opções no método `match_and_run()` em `main.py`.

Para rodar a pipeline padrão do projeto: `python main.py pipeline`.

## Resultados
Os resultados são gerados pelo parâmetro `pipeline_results` ou `pipeline` para pasta `output/`. Após gerados, os resultados
devem ser carregados na pasta do Jupyter com o parâmetro `output_load_jupyter`.

Após carregados, entrar na pasta `notebook/` (fora do shell do Poetry) e executar `pip install` e em seguida `jupyter notebook` e executar `pip install` e em seguida `jupyter notebook`.
Isso ira executar o Jupyter Notebook, e é possível examinar os resultados pelos arquivos correspondentes na pasta.

### Dados
Extrair arquivo `Data.tar.gz` para a pasta `Data/` para obter os dados básicos necessários para rodar os notebooks.
Novos dados podem ser obtidos ao rodar o programa para gerar novos resultados.

O arquivo pode ser obtido em [Data.tar.gz](https://drive.google.com/file/d/1JzvEEENyiN5h6VaU7gnQCzazBdwOmpOi/view?usp=share_link)
