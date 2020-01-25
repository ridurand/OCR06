# -*- coding: utf-8 -*-
"""
STACKOVERFLOWTAGS_RECSYS_DASH
Application simple qui propose une liste de tags StackOverflow relatifs 
à une question saisie traitant de sujets informatiques

Pickles (.pkl) nécessaires : 
- ignore_words (préprocessing) : liste des mots qui ne doivent pas être modifiés par le NLP
- specialtags (préprocessing) : tags contenant des caractères spéciaux (C#...)
- manual_stopwords (préprocessing) : stopwords issus de l’analyse exploratoire
- mlb (préprocessing) : multilabelbinarizer pour transformer les prédictions supervisées en libellé
- tf_unsupervised (préprocessing) : transformer TF pour l’approche non supervisée
- tfidf_supervised (préprocessing) : transformer TFIDF pour l’approche supervisée
- lda_model (recommandation) : modèle non supervisé 
- lr_top100tags_3labels (recommandation) : modèle supervisé

A exécuter dans stackoverflowtags_recsys_dash 

exemple de phrases : 
I want to write a simple regular expression in Python that extracts a number from HTML.
This sql request grouping values by keys on the relational database is not working.    
I want to develop a web application generating html, javascript and css, what is the good language to do that.
I want to code a Python function to sum item from a dictionary.
    
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import pickle 
from utils import clean_whitespace_and_code, apply_specialtags_transco, clean_punctuation, stopWordsRemove, lemmatization, pred_nwords_unsupervised, recommend_tags
#heroku:from .utils import clean_whitespace_and_code, apply_specialtags_transco, clean_punctuation, stopWordsRemove, lemmatization, recommend_tags

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

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
      result = recommend_tags(value, 5, seuil=0.22, clean=True)
      supervised = result['Supervised'][0]
      unsupervised = result['Unsupervised'][0]
    return supervised, unsupervised 

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
#    app.run_server()
