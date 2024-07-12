import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def numerical_engineering(new_score): 
    new_score['age'] = new_score['age'].astype(float)
    new_score['age'] = new_score['age'].apply(lambda x: int('1' + str(int(abs(x)))) if x < 10 else int(x))

    new_score['num_students'] = new_score['n_male'] + new_score['n_female']
    return new_score 

def categorical_engineering(new_score): 
    new_score['number_of_siblings'] = new_score['number_of_siblings'].astype('str')
    new_score['attendance_status'] = new_score['attendance_rate'].apply(lambda x: 'High' if x >= 90 else 'Low')

    CCA_mapping = {'CLUBS': 'Clubs', 'ARTS': 'Arts', 'SPORTS': 'Sports', 'NONE': 'None'}
    tuition_mapping = {'Y': 'Yes', 'N': 'No'}
    
    new_score = new_score.replace({"CCA": CCA_mapping})
    new_score = new_score.replace({"tuition": tuition_mapping})
    return new_score

def time_engineering(new_score): 
    new_score['sleep_time_status'] = new_score['sleep_time'].apply(lambda x: 'Before 11.30pm' if x in ['22:00', '22:30', '21:00', '21:30', '23:00'] else '11:30pm and later')
    new_score['sleep_time'] = pd.to_datetime(new_score['sleep_time'], format='mixed') 
    new_score['wake_time'] = pd.to_datetime(new_score['wake_time'], format='mixed') 
    new_score['hours_sleep'] = new_score.apply(lambda row: (row['wake_time'] - row['sleep_time']).total_seconds() / 3600 if row['wake_time'] > row['sleep_time'] else ((row['wake_time'] + pd.Timedelta(days=1)) - row['sleep_time']).total_seconds() / 3600, axis=1)
    new_score['sleep_hour_status'] = new_score['hours_sleep'].apply(lambda x: '6 and below' if x <= 6.0 else 'Above 6')
    print(new_score.head())
    return new_score

def filter_features(new_score): 
    new_score = new_score[['direct_admission', 'CCA', 'number_of_siblings', 'learning_style', 'tuition', 'final_test', 'num_students', 'age', 
                            'hours_per_week', 'attendance_status', 'sleep_time_status', 'sleep_hour_status']]
    return new_score


class EncoderScaling: 
    def __init__(self, dataframe): 
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.df = dataframe
        self.X_train = None 
        self.X_test = None 
        self.y_train = None
        self.y_test = None 
    
    def train_test_split(self): 
        X = self.df.drop('final_test', axis=1)
        y = self.df['final_test']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Shape of X_train: {self.X_train.shape}")
        print(f"Shape of X_test: {self.X_test.shape}")

    def categorical_encoder(self): 
        categorical = ['direct_admission', 'number_of_siblings', 'CCA', 'learning_style', 'tuition', 'attendance_status', 'sleep_time_status', 'sleep_hour_status']
        encoder_dict = {}

        for feature in categorical: 
            self.encoder.fit(self.X_train[[feature]])
            encoder_dict[feature] = self.encoder
            
            encoded_train = self.encoder.transform(self.X_train[[feature]])
            encoded_test = self.encoder.transform(self.X_test[[feature]])
            encoded_feature_names = self.encoder.get_feature_names_out([feature])
            
            encoded_train = pd.DataFrame(encoded_train, columns=encoded_feature_names, index=self.X_train.index)
            encoded_test = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=self.X_test.index)
            
            self.X_train = self.X_train.drop(columns=[feature])
            self.X_test = self.X_test.drop(columns=[feature])
            
            self.X_train = pd.concat([self.X_train, encoded_train], axis=1)
            self.X_test = pd.concat([self.X_test, encoded_test], axis=1)
    
    def scaling(self): 
        self.scaler.fit(self.X_train)
        
        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.X_train.columns, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.X_test.columns, index=self.X_test.index)
    
def feature_engineering(new_score): 
    df = numerical_engineering(new_score)
    df = categorical_engineering(df)
    df = time_engineering(df)
    df = filter_features(df)

    encode_and_scale = EncoderScaling(df)
    encode_and_scale.train_test_split()
    encode_and_scale.categorical_encoder()
    encode_and_scale.scaling()

    print(encode_and_scale.X_train.head())
    return encode_and_scale.X_train, encode_and_scale.X_test, encode_and_scale.y_train,encode_and_scale.y_test

    





