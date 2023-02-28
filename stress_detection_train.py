

###Imports

import pandas as pd
import uuid
import pickle
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import time

# # Reading Input data

def generate_uuids(n):
    ids=[]
    for i in range(n):
        ids.append(str(uuid.uuid4()))       
    return ids

def read_dataFrame(input_file):
    df=pd.read_csv(input_file)
    df.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen','eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']
    df['id']=generate_uuids(len(df))
    X = df.drop(['snoring_rate','limb_movement','eye_movement','stress_level','id'],axis=1)
    y = df.stress_level
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7, 
                                                    random_state=0)
    return X_train,y_train,X_test,y_test


# # Model Training and logging to MLflow



def training(X_train,y_train,X_test,y_test):
    clf = GradientBoostingClassifier(n_estimators=20, random_state = 0)
    model= clf.fit(X_train, y_train)   
    pickle.dump(model,open('model.pkl','wb'))
    y_pred = clf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred,squared="False")  
    print("RMSE of Gradient boost is: %.4f" % rmse)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy of Gradient boost is ", accuracy)



    
def train():
    input_file='./data/SaYoPillow.csv'
    X_train,y_train,X_test,y_test=read_dataFrame(input_file)
    time.sleep(3)
    training(X_train,y_train,X_test,y_test)
    print("Finished")
    



if __name__ == '__main__':
    train()

