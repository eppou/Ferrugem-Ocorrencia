import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from testlab.constants import INPUT_PATH


def run():
    path = INPUT_PATH / "aluguel.csv"

    imoveis = pd.read_csv(path, sep=";")
    imoveis["Valor Total"] = (
        parse(imoveis["Valor"]) + parse(imoveis["Condominio"]) + parse(imoveis["IPTU"])
    )

    datas_pesquisa = []
    current_date = pd.Timestamp("2018-01-01")
    for d in range(imoveis.shape[0]):
        datas_pesquisa.append(current_date)

        if d != 0 and (d % 5) == 0:
            current_date = current_date + DateOffset(days=1)

    imoveis["Data Pesquisa"] = datas_pesquisa

    for index, imovel in imoveis.iterrows():
        cd = imovel["Data Pesquisa"]
        cd_7d = cd - DateOffset(days=7)
        cd_14d = cd - DateOffset(days=14)

        valor_total_dia = imoveis[imoveis["Data Pesquisa"] == cd]["Valor Total"].sum()
        valor_total_7d = imoveis[
            (imoveis["Data Pesquisa"] <= cd) & (imoveis["Data Pesquisa"] > cd_7d)
        ]["Valor Total"].sum()
        valor_total_14d = imoveis[
            (imoveis["Data Pesquisa"] <= cd) & (imoveis["Data Pesquisa"] > cd_14d)
        ]["Valor Total"].sum()

        imoveis.at[index, "Valor Total - Dia"] = valor_total_dia
        imoveis.at[index, "Valor Total - Últimos 7d"] = valor_total_7d
        imoveis.at[index, "Valor Total - Últimos 14d"] = valor_total_14d

    print(imoveis.head(n=200))
    # print(imoveis.describe())
    # print(imoveis.dtypes)


def parse(val: pd.Series):
    return val.replace(np.nan, 0)
