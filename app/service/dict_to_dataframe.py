import pandas as pd


def dict_to_dataframe(diccionario: dict):
    diccionario['Genre (Mechanic)'] = diccionario['genre_mechanic']
    diccionario['Component'] = diccionario['component']
    diccionario['Theme definition'] = diccionario['theme_definition']

    return pd.DataFrame({k: [v] for k, v in diccionario.items()},
                        columns=['name', 'min_players', 'max_players',
                                 'playing_time', 'min_age', 'families',
                                 'mechanic', 'Domain', 'Genre (Mechanic)', 'Theme definition',
                                 'Component', 'weight_average', 'L_B',
                                 'L_M', 'L_A', 'desv_jg'])
