# -*- coding: utf-8 -*-
"""
STACKOVERFLOW_PREDICT_DASH
Application simple qui propose une liste de tags StackOverflow relatifs 
à une question saisie traitant de sujets informatiques

Pickles (.pkl) nécessaires : 
    - top500tags (préprocessing) : contient les 500 tags StackOverflow les plus fréquents du jeu d'entraînement
    - mlb (préprocessing) : multilabelbinarizer pour transformer les prédictions supervisées en libellé
    - tfidf (préprocessing) 
    - lda_model_list (recommandation) : modèle non supervisé 
    - lr_top100tags_3labels (recommandation) : modèle supervisé 

A exécuter dans stackoverflow_predict_dash

exemple de phrases : 
This sql request grouping values by keys on the relational database is not working.    
I want to develop a web application generating html, javascript and css, what is the good language to do that.
I want to code a Python function to sum item from a dictionary.
    
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pickle 
import numpy as np
from utils import clean_text, clean_punctuation, stopWordsRemove, lemmatization, Recommend_tags

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# CHARGEMENT
with open('top500tags.pkl', 'rb') as f:
    top500tags = pickle.load(f)    
with open('lda_model_list.pkl', 'rb') as f:
    lda_model_list = pickle.load(f)    
with open('lr_top100tags_3labels.pkl', 'rb') as f:
    lr_top100tags_3labels = pickle.load(f) 
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f) 
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)     

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(children=[
    html.H2("Recommandation de tags StackOverFlow pour une question"),
    html.Div(dcc.Input(id='input_textori',
                       style={'width': '800px'},
                       type='text',
                       placeholder='Saisir une question'), style={'marginBottom': 10, 'marginTop': 10}),        
    html.Button(id='submit-button', n_clicks=0, type='submit', children='Submit'),
    html.Div(dcc.Input(id='output_tagssupervised',
                       style={'width': '800px'},                       
                       type='text',
                       placeholder="Tags issus de l'analyse supervisée"), style={'marginBottom': 10, 'marginTop': 10}), 
    html.Div(dcc.Input(id='output_tagsunsupervised',
                       style={'width': '800px'},                       
                       type='text',
                       placeholder="Tags issus de l'analyse non supervisée"), style={'marginBottom': 10, 'marginTop': 10})    
    ])

@app.callback([dash.dependencies.Output('output_tagssupervised', 'value'),
               dash.dependencies.Output('output_tagsunsupervised', 'value')], 
              [dash.dependencies.Input('submit-button', 'n_clicks')],
              [dash.dependencies.State('input_textori', 'value')])
def update_output(n_clicks, value):
    supervised = ''
    unsupervised = ''
    if n_clicks > 0:
      result = Recommend_tags(value, 
                              5, 
                              mlb, 
                              tfidf, 
                              lda_model_list[4], 
                              lr_top100tags_3labels.best_estimator_, 
                              seuil=0.22, 
                              clean=True)
      supervised = result['Supervised'][0]
      unsupervised = result['Unsupervised'][0]
    return supervised, unsupervised 

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
