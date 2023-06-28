from pprint import pprint

import pandas as pd


def shapefile_to_dataframe(sf):
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]

    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)

    return df


def run():
    sf = load()
    df = shapefile_to_dataframe(sf)

    pprint(df.shape)
    pprint(df.columns)
    # pprint(df.sample(5))
    pprint(df[['nome', 'coords']])
