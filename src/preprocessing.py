import pandas as pd 
import numpy as np

def drop_duplicates(score_df): 
    print(f"There are {score_df.shape[0]} rows and there are {score_df['student_id'].nunique()} unique students in score_df.")

    # Drop bag column and remove duplicates since most duplicate student ID records occur due to change in bag color. 
    # bag color has no influence on test scores. 
    score_df = score_df.drop(['bag_color', 'index'], axis = 1)
    score_df = score_df.drop_duplicates()
    print(f"After dropping the duplicate rows, we have {score_df.shape[0]} rows.")

    # Drop rows with null values for either attendance rate or final test score since they contributed to duplicates.
    new_score = score_df.dropna(subset = ['attendance_rate', 'final_test'], how = 'any')
    new_score = new_score.drop_duplicates()
    print(f"There are {new_score.shape[0]} rows after removing all duplicates.")

    return new_score


def deal_with_null(score_df, new_score): 
    null_features = [feature for feature in score_df.columns 
                 if score_df[feature].isnull().sum() > 1]
        
    for feature in null_features: 
        print(f"{feature} has {score_df[feature].isnull().sum()} missing values (ie. {round(score_df[feature].isnull().sum() / len(score_df) * 100.0,2)}% of dataset).")

    null_features = [feature for feature in new_score.columns 
                 if new_score[feature].isnull().sum() > 1]
    
    print("After removal of duplicates: ")
    if len(null_features) > 0: 
        for feature in null_features: 
            print(f"{feature} has {new_score[feature].isnull().sum()} missing values (ie. {round(new_score[feature].isnull().sum() / len(new_score) * 100.0,2)}% of dataset).")
    else: 
        print("There are no null features.")