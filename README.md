# NLP-Disaster-Classification
In this project, disaster data from Figure Eight is analyzed to build a model for an API that classifies disaster messages. The project includes three components.

## 1.	ETL Pipeline
The first part of the project is a code titled “process_data.py” in the data folder, which is a ETL pipeline for cleaning and transforming the data set to make it appropriate for modeling. The final data set is stored in the “DisasterResponse” SQLite database. 

## 2. ML Pipeline
The second component of the project, the machine learning aspect, is located in the models folder. The final model is built that uses the message column of the data set to predict classifications for 36 categories using a machine learning pipeline that includes NLTK, scikit-learn's Pipeline and GridSearchCV (multi-output classification). At the end the model is exported to a pickle file named “classifier.pkl”. The code for training the model is saved as “train_classifier.py”.

## 3. Flask Web App 
A web app is being developed for the third stage of the project, where an emergency worker may enter a new message and receive categorization results in numerous categories. The data visualizations are also displayed in the web app. 

## Instructions:
1.	Run the following commands in the project's root directory to set up the database and model.
•	To run ETL pipeline (process_data.py) that cleans data and stores in database titled “DisasterResponse.db”:

“python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db”

•	To run ML pipeline (train_classifier.py) that trains classifier and saves model as “classifier.pkl”:

“python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl”

This code will take hours to run because of long list f the parameters for grid search. To examine the code faster, you can reduce the parameters.

2.	To run the Web App from the Project Workspace IDE, open a new terminal window. Type in the command line:

“python app/run.py”

The web app should now be running if there were no errors. Now, open another Terminal Window and type:

“env|grep WORK”

You'll see output that looks something like this:

 

In a new web browser window, type in the following:

https://SPACEID-3001.SPACEDOMAIN

In this example, that would be: "https://viewa7a4999b-3001.udacity-student-workspaces.com/" (Don't follow this link now, this is just an example.)
The SPACEID might be different.
You should be able to see the web app. The number 3001 represents the port where your web app will show up. Make sure that the 3001 is part of the web address you type in.


The web page looks like this: 
















If you write your message there, it will classify the message. The page includes three data visualization

