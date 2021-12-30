# NLP-Disaster-Classification
In this project, disaster data from Figure Eight is analyzed to build a model for an API that classifies disaster messages. The project includes three components.

## 1.	ETL Pipeline
The first part of the project is a code titled “process_data.py” in the data folder, which is a ETL pipeline for cleaning and transforming the data set to make it appropriate for modeling. The final data set is stored in the “DisasterResponse” SQLite database. 

## 2. ML Pipeline
The second component of the project, the machine learning aspect, is located in the models folder. The final model is built that uses the message column of the data set to predict classifications for 36 categories using a machine learning pipeline that includes NLTK, scikit-learn's Pipeline and GridSearchCV (multi-output classification). At the end the model is exported to a pickle file named “classifier.pkl”. The code for training the model is saved as “train_classifier.py”.

## 3. Flask Web App 
A web app is being developed for the third stage of the project, where an emergency worker may enter a new message and receive categorization results in numerous categories. The data visualizations are also displayed in the web app. 

