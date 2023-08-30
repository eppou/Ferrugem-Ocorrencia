from testlab.constants import INPUT_PATH
import pandas as pd


def run():
    path = INPUT_PATH / "aluguel.csv"

    imoveis = pd.read_csv(path, sep=";")

    imoveis_tipo = imoveis["Tipo"].drop_duplicates()
    print(type(imoveis_tipo))
    print(imoveis_tipo.count())
