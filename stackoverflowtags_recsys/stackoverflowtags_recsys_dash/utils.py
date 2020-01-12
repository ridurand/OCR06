import numpy as np
import pandas as pd
import pickle 
import re, spacy, nltk
#heroku:import en_core_web_md
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation

with open('./stackoverflowtags_recsys_dash/top500tags.pkl', 'rb') as f:
    top500tags = pickle.load(f)    

#heroku:nlp = en_core_web_md.load()
nlp = spacy.load('en', disable=['parser', 'ner'])   
spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)

def clean_text(text):
    ''' Lowering text and removing undesirable marks
    Parameter:
    text: corpus to clean
    '''

    
    text = text.lower()
    text = re.sub(r"\'\n", " ", text) # removes line feeds
    text = re.sub(r"\'\xa0", " ", text) # removes spaces
    text = re.sub('\s+', ' ', text) # matches all whitespace characters : \t\n\r\f\v
    text = text.strip(' ') # removes leading and trailing blanks
    
    return text

def clean_punctuation(text, ignore_words): 
    ''' Remove punctuation
    Parameters:
    text: corpus to remove punctuation from it
    ignore_words: list of words to include without processing them
    '''
    
    
    words = spacy_tokenizer(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punctuation))
    
    for w in words:
        # certains mots peuvent être des tags qui contiennent des signes de ponctuation, il faut les conserver tels quels
        # en utilisant une liste de top tags la plus exhaustive possible
        if str(w) in ignore_words: 
            punctuation_filtered.append('<' + str(w) + '>')
        else:
            w = re.sub('[0-9]', ' ', str(w)) # word contains no digits
            punctuation_filtered.append(regex.sub(' ', str(w)))
        
    result = ' '.join(punctuation_filtered)
    result = re.sub(' +', ' ', result) # remove duplicates whitespaces
    
    return result 

def stopWordsRemove(text, stop_words):
    ''' Removing all the english stop words from a corpus
    Parameter:
    text: corpus to remove stop words from it
    stop_words: list of stop words to exclude
    '''

    
    words = spacy_tokenizer(text)
    filtered = [str(w) for w in words if not str(w) in stop_words]
    text = ' '.join(map(str, filtered))
    result = re.sub(' +', ' ', text) # remove duplicates whitespaces
    
    return result

def lemmatization(text_in, allowed_postags, ignore_words):
    ''' It keeps the lemma of the words (lemma is the uninflected form of a word),
    and deletes the undesired POS tags
    
    Parameters:
    
    text_in (list): text to lemmatize
    allowed_postags (list): list of allowed postags, like NOUN, ADJ, VERB, ADV
    ignore_words: list of words to include without processing them
    '''

    
    doc = nlp(text_in) 
    text_out = []
    
    for token in doc:
        
        if str(token) in ignore_words:
            text_out.append(str(token))
            
        elif token.pos_ in allowed_postags:            
            text_out.append(token.lemma_)
                
    text_out = ' '.join(text_out)
    result = re.sub(' +', ' ', text_out) # remove duplicates whitespaces

    return result

def Recommend_tags(text_ori, n_words, mlb, tfidf, lda, clf, seuil=0.5, toptags=top500tags, clean=False):
    
    ''' Recommendation system for StackOverflow posts based on a lda model, it returns up to 5 words.

    Parameters:

    text: the stackoverflow post of user
    n_words: number of tags to recommend
    seuil: threshold for decision
    clean: True if data preparation is needed
    '''

    auto_stopwords = set(set(nlp.Defaults.stop_words) | set(stopwords.words("english")))
    manual_stopwords = ['file', 'way', 'application', 'user', 'use', 'method',
                        'example', 'problem', 'work', 'test', 'question', 'project', 'thank',
                        'return', 'solution', 'thing', 'change', 'program', 'idea', 'end',
                        'message', 'result', 'answer', 'issue', 'language', 'information',
                        'document', 'e', 'default', 'help', 'people', 'run', 'testing',
                        'difference', 'stuff', 'need', 'response', 'check', 'product', 'approach', 
                        'want', 'suggestion', 'n', 'custom', 'create', 'p', 'place', 'comment',
                        'support', 'person', 'expression']
    manual_stopwords = set(manual_stopwords)    
    
    if type(text_ori) in (str, pd.Series):
        if type(text_ori) is str:
            text_ori = pd.Series(text_ori) 
        text = text_ori
        text_ori = text_ori.rename("Texte d'origine")
        text = text.rename("Texte modifié")
    else:
        return 'Type should be str or pd.Series'

    if clean==True:
        text = text.apply(lambda x: clean_text(x))
        text = text.apply(lambda x: BeautifulSoup(x, features="lxml").get_text())
        text = text.apply(lambda x: clean_punctuation(x, top500tags))        
        text = text.apply(lambda x: stopWordsRemove(x, auto_stopwords))
        text = text.apply(lambda x: lemmatization(x, ['NOUN'], top500tags))   
        text = text.apply(lambda x: stopWordsRemove(x, manual_stopwords))

    # document = question StackOverflow
    # word = il s'agit des mots issus du vocabulaire LDA, retenus par le LDA (max_features)
    # topic = il s'agit des topics issus du LDA (components)
    # pour chaque document et chaque mot, je calcule la probabilité totale qu'un mot apparaisse dans le document 
    # proba(word1) = proba(word1/topic1) * proba(topic1) + proba(word1/topic2) * proba(topic2) ...
    # je conserve les n_tags
    
    document_tfidf = tfidf.transform(text)
    proba_topic_sachant_document = lda.transform(document_tfidf)
    word_labels = tfidf.get_feature_names()
    proba_word_sachant_topic = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis] # normalization    
    #print(proba_topic_sachant_document.shape)    
    #print(proba_word_sachant_topic.shape)    
    
    # proba_topic_sachant_document est de dimension d x t
    # proba_word_sachant_topic est de dimension t x w
    # je peux donc opérer un produit matriciel entre les 2 matrices pour calculer pour chaque document : proba(wordn)
    # j'obtiendrai une matrice proba_word_sachant_document de dimension d x w
    # il ne me restera plus qu'à choisir les "n_words" mots les plus probables
    proba_word_sachant_document = proba_topic_sachant_document.dot(proba_word_sachant_topic)  
    
    # je transforme la matrice en dataframe : 
    # data = les proba des mots pour chaque document
    # index = l'index des données en entrée
    # columns = les labels des mots sélectionnés en sortie du LDA
    df_wd = pd.DataFrame(data=proba_word_sachant_document,
                         index=text.index,
                         columns=word_labels) 
    
    # np.argsort(-df_wd.values, axis=1)[:, :n_words])
    # renvoie pour chaque document, les "n_words" indexes des colonnes dont les proba sont les plus élevées
    # grâce aux indexes, je peux récupérer le libellé de la colonne qui est donc le libellé du mot 
    # et le stocker en ligne
    values = df_wd.columns.values[np.argsort(-df_wd.values, axis=1)[:, :n_words]]
    values = [", ".join(item) for item in values.astype(str)]
    #pred_unsupervised = pd.DataFrame(df_wd.columns.values[np.argsort(-df_wd.values, axis=1)[:, :n_words]],
    #                                 index=df_wd.index,
    #                                 columns = ['word' + str(i + 1) for i in range(n_words)])
    pred_unsupervised = pd.DataFrame(values,
                                     index=df_wd.index,
                                     columns = ['Unsupervised'])
    
    pred_supervised = pd.DataFrame(clf.predict_proba(tfidf.transform(text))).applymap(lambda x:1 if x>seuil else 0).to_numpy()
    pred_supervised = pd.Series(mlb.inverse_transform(pred_supervised), name='Supervised', index=text.index)
    pred_supervised = pred_supervised.apply(lambda row: ', '.join(row))
    result = pd.concat([pred_supervised, pred_unsupervised, text_ori, text], axis=1)
    
    return result

