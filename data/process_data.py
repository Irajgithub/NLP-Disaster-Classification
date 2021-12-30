# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the messages and categories files,
    removes duplication and returns merged data frame.
    Input: 2 csv files path
    Output: Pandas dataframe of the merged csv files
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    messages.drop_duplicates(subset='id',keep='first', inplace=True)
    categories.drop_duplicates(subset='id', keep='first', inplace=True)
    
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    '''
    This function rearranges the columns to create 
    distinct columns for each category.
    Input: Pandas dataframe
    Output: Pandas dataframe incloding individual category columns
    '''
    
    df_split = df['categories'].str.split(';', expand = True)
    
    # rename each category column
    i=0
    while i<36:
        name=df_split.loc[0,i].split('-')[0]
        df_split.rename(columns={df_split.columns[i]:name}, inplace=True)
        i+=1
    
    # clean the contents of the category columns and convert to int
    for column in df_split:
        df_split[column] = df_split[column].str.split('-').apply(lambda x:x[-1])
        df_split[column] = df_split[column].astype(int)
    
    df.drop('categories',axis=1, inplace=True)
    df_clean = pd.concat([df,df_split], axis=1)
    
    return df_clean


def save_data(df, database_filename):
    '''
    This function saves the data fram as a SQL file.
    Input: Pandas dataframe
    Output: No return, saved SQL file
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterTable_messages', engine, index=False)
     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()