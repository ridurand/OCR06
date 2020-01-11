# -*- coding: utf-8 -*-
"""
ARRDELAY_PREDICT_DASH
Application web affichant les vols dans un créneau : 
    - 1h avant leur heure de départ prévue
    - 2h après leur heure d'arrivée prévue
Pour les vols ayant décollé, un retard estimé est calculé
Les autres sont marqués "Non décollé"

Pickles (.pkl) nécessaires : 
    - df_dash : jeu de données simulant des vols sur 2019
    - mapper_fit : fonction de transformation des features nécessaire au calcul des prédictions
    - ridge_reg : fonction prédictive issue de la régression ridge qui va me permettre de calculer les prédictions

A exécuter dans arrdelay_predict_dash
    
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import pickle as pkl
import numpy as np
import datetime as dt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

now = dt.datetime.now()

# importation du jeu de données simulant des vols sur 2019
df_dash = pd.read_pickle('./arrdelay_predict_dash/df_dash.pkl')
df_today = df_dash[pd.to_datetime(df_dash['FL_DATE']).dt.date == now.date()]

# (CRS_DEP_TIME - 1h < heure courante) 
BORNE_INF = df_dash['CRS_DEP_TIME'].apply(lambda t : (dt.datetime.combine(dt.date(2019, 1, 1), t) - dt.timedelta(hours=1)).time())
# (heure courante < CRS_ARR_TIME + 2h) 
BORNE_SUP = df_dash['CRS_ARR_TIME'].apply(lambda t : (dt.datetime.combine(dt.date(2019, 1, 1), t) - dt.timedelta(hours=2)).time())

df_today = df_today.loc[(BORNE_INF <= now.time()) & (BORNE_SUP >= now.time())]

# combinaison code de l'aéroport / nom de l'aéroport pour alimenter la liste déroulante de l'aéroport de destination à sélectionner
# ce sont les aéroports accueillant des vols correspondant aux critères de sélection ci-dessus
dest_airport_name = dict(zip(df_today.DEST_AIRPORT_ID, df_today.DEST_AIRPORT_LIB))

X_var = ['DEP_DELAY','DEST_AIRPORT_NB_FLIGHTS_BY_DAY', 'UNIQUE_CARRIER', 'DEST_TRAFIC_AIRPORT']
#df_pred = df_today.loc[df_today['WHEELS_OFF'] <= now.time(), X_var]
df_pred = df_today[X_var]

# Importation de la fonction de transformation des features nécessaire au calcul des prédictions
mapper_fit = pd.read_pickle('./arrdelay_predict_dash/fitted_mapper.pkl')
# Importation de la fonction prédictive issue de la régression ridge qui va me permettre de calculer les prédictions
ridge_reg = pd.read_pickle('./arrdelay_predict_dash/ridge_reg.pkl')

# Reconstitution des features nécessaires au calcul des prédictions
df_pred = np.round(mapper_fit.transform(df_pred.copy()), 2)
# Calcul des prédictions 
df_today['PREDICTION'] = np.round(ridge_reg.predict(df_pred),2)

# Pour les avions qui n'ont pas encore décollé, on annule la prédiction
# En pratique, je n'utiliserais pas un tel subterfuge car WHEELS_OFF ne serait pas pré-renseigné 
# comme dans mon jeu simulé mais l'information devrait parvenir en temps réel 
df_today.loc[df_today['WHEELS_OFF'] > now.time(), 'PREDICTION'] = 'Non décollé'

columns_toprint = ['FL_NUM', 'ORIGIN_AIRPORT_LIB', 'LIB_CIE', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'PREDICTION']
columns_toprint_lib = ['Numéro de vol', "Aéroport d'origine", 'Compagnie', 'Heure de départ prévue', "Heure d'arrivée prévue", 'Retard estimé (en minutes)']
columns_todash = [{"name": j, "id": i} for i, j in zip(columns_toprint, columns_toprint_lib)]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(children=[
    html.H2("Prévisions des retards à l'arrivée de vols"),
    html.Div("Nous sommes le {:%d/%m/%Y à %H:%M}".format(now), style={'marginBottom': 10, 'marginTop': 10, 'color': 'red', 'fontSize': 20}), 
    # Application simple proposant une liste déroulante (=dpdn_dest_airport)
    # Quand on sélectionne une valeur dans cette liste, cela met à jour le tableau des vols (=output_table) 
    # qui entrent dans le créneau défini en amont 
    html.Div("Les vols sont affichés 1h avant leur heure de départ prévue et jusqu'à 2h après leur heure d'arrivée prévue", style={'marginBottom': 10, 'marginTop': 10, 'color': 'red', 'fontSize': 16}),     
    html.Div(children=[
            html.Div('Liste des destinations des vols'),
            dcc.Dropdown(
                    id='dpdn_dest_airport',
                    style={'width': '800px'},
                    options=[{'label': v, 'value': k} for k, v in dest_airport_name.items()],
                    value=10397,
                    placeholder="Sélectionnez un aéroport de destination",
                    )
            ], style={'marginBottom': 10, 'marginTop': 10}),        
    html.Button(id='submit-button', n_clicks=0, type='submit', children='Submit'),
    html.Div(dash_table.DataTable(id='output_table',
                                  data=[],
                                  columns=columns_todash
                                 ), style={'marginBottom': 10, 'marginTop': 10})
])

@app.callback([dash.dependencies.Output('output_table', 'data'),
               dash.dependencies.Output('output_table', 'columns')], 
              [dash.dependencies.Input('submit-button', 'n_clicks')],
              [dash.dependencies.State('dpdn_dest_airport', 'value')])
def update_output(n_clicks, input1):
    if n_clicks is None:
        data = []
    elif n_clicks == 0:
        data = []
    else:
        df_maj = df_today.loc[df_today['DEST_AIRPORT_ID'] == input1, columns_toprint]    
        df_maj = df_maj.sort_values(by=['LIB_CIE', 'FL_NUM'])
        if df_maj.empty:
            data = []
        else:
            data = df_maj[columns_toprint].to_dict('records')  

    return data, columns_todash
    
if __name__ == '__main__':
    app.run_server()
