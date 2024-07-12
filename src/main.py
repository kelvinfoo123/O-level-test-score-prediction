import sqlite3
import pandas as pd 
import preprocessing
import feature_engineering
import models 

def main(): 
    conn = sqlite3.connect(r"/Users/kelvinfoo/Desktop/AISG Technical Assignments/Test Score Prediction/score.db")
    df = pd.read_sql_query("SELECT * FROM score", conn)
    new_score = preprocessing.drop_duplicates(df)
    preprocessing.deal_with_null(df, new_score)
    X_train, X_test, y_train, y_test = feature_engineering.feature_engineering(new_score)
    models.run_model(X_train, X_test, y_train, y_test)

if __name__ == '__main__': 
    main()