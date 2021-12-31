# import libraries
import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import sqlite3

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    This function loads data from SQL 
    and returns X , Y for training the model
    Input: database_filepath: database file path
    Output: X (traing message list), Y (training target), category names
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterTable_messages', engine)
    
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    
    category_names = list(Y.columns)
    
    return X, Y, category_names


def get_wordnet_pos(word):
    '''Map pos tag to first character lemmatize() accepts'''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def customized_stop_words():
    '''
    This function customize the stop-words list for tokenizing 
    that is appropriate for the classification model
    Input: no input
    Output: customized stop-words list  
    '''
    
    stop_words = stopwords.words('english')
    
    new_stopwords = ['could','may', "might",'please','would','like','even','thanks','thank','good','nice','fine','well',\
                     'bless','god','bad','badly','today','night','tonight','morning','evening','noon',\
                     'afternoon','tomorrow','fewer','little','less','several','many','much','last','least','end','early',\
                     'late',\
                     'final','finally','formal','formally','informal','informally','already','regard','regarding','according',\
                     'following','now','every','evryone','everything','anyone','everyone','anything','everything','any',\
                     'someone','something','thing','nothing','due','since','message',\
                     'messages','text','problem','problems','information','report','reports','people','link','tendency','know',\
                     ' knew','known','knowing','knows','say','said','sayed','saying','says','tell','told','tells','telling',\
                     'talk','talked','talks','talking','know','knew','known','knows','knowing','want','wanted','wants',\
                     'let','lets','letting','need','needed','needs','needing','help','helped','helping','helps','see','saw',\
                     'seen','seeing','receive','received','receiving','receives','ask','asked','asking','asks','get','got',\
                     'gotten','gets','getting','give','gave','given','gives','giving','take','took','taken','takes',\
                     'taking','us',\
                     'respect','respectively','respective']
    
    grouped_stopwords = ['day','days','month','months','week','weeks','year','years','number','numbers','numberst',\
                         'numbernd','numberrd','numberth','numberteen','numberty']
    
    stop_words = stop_words + new_stopwords + grouped_stopwords
    
    return stop_words


def tokenize(text):
    '''
    Tokenizing of a text data to normalize, lemmatize, stem and tokenize the text. 
    Input: Text data
    Output: List of clean tokens 
    '''
    
    text_token_clean =[]
    text_token_clean2 =[]
    text_token_clean3 =[]
    text_token_clean4 =[]
    
    day_group = '|'.join(['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])
    
    number_group = '|'.join(['first','second','third','eleven','twelve','thirteen','fourteen','fifteen','sixteen',\
                             'seventeen','eighteen','nineteen','twenty','thirty','forty','fifty','sixty','seventy',\
                             'eighty','ninety','one','two','three','four','five','six','seven','eight','nine','ten','hundred'])

    month_group = '|'.join(['january','jan','february','feb','march','mar','april','apr','may','june','jun','july','jul',\
                            'august','aug','september','sep','sept','october','oct','november','nov','december','dec'])
    
    # normalizing the text by lowering the cases, converting all numerical and alphabetical values to the word 'number',
    # converting all weekdays to the word 'day', all month names to the word 'month', and finally removing symbols.
    text = text.lower()
    text = re.sub('(?<=\d)[,-./](?=\d)', '', text)
    text = re.sub(r'\d+','number', text)
    text = re.sub(number_group,'number', text)
    text = re.sub(day_group, 'day', text)
    text = re.sub(month_group, 'month', text)
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    
    text_token = word_tokenize(text)

    # removing stop words
    for w in text_token:
        if w not in customized_stop_words():
            text_token_clean.append(w)

    # converting the words to their root by WordNetLemmatizer function
    for w in text_token_clean:
        text_token_clean2.append(WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)))

    # converting the words to their root by PorterStemmer function
    for w in text_token_clean2:
        text_token_clean3.append(PorterStemmer().stem(w))
     
    # checking stop words again
    for w in text_token_clean3:
        if w not in customized_stop_words():
            text_token_clean4.append(w)
    
    return text_token_clean4

def build_model():
    '''
    Build Machine learning pipleine using svm classifier
    Input: None
    Output: clf gridSearch Model
    '''
    
    pipeline = Pipeline([
        ('vect' , CountVectorizer(tokenizer=tokenize)),
        ('tfidf' , TfidfTransformer()),
        ('clf' , MultiOutputClassifier(svm.SVC()))
    ])
    
    # grid search parameters
    parameters = {
    'tfidf__norm':['l2', 'l1'], 
    'clf__estimator__C' : [10, 1.0, 0.1],
    'clf__estimator__kernel' : ['linear', 'poly', 'rbf'],
    'clf__estimator__degree' : [3, 4] 
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints the test report for the model
    Input: model (trained model), X_test (test data), Y_test (true test labels for the X_test)
    Output: None 
    '''
    
    Y_pred = model.predict(X_test)
    
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Export the model as a pickle file
    Input: model (trained model), model_filepath (location to store the model)
    Output: None
    '''
    
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()