# O-level-test-score-prediction
**Overview**: Prediction of student's O Level test score.

### **Workflow**
<img width="1014" alt="Screenshot 2024-06-24 at 11 12 00" src="https://github.com/kelvinfoo123/O-level-test-score-prediction/assets/112041340/3a0c206e-a683-45d8-a343-bc950a5e2c5b">

### **1. Preprocessing**
1. Null features (test score and attendance), rows with duplicate student ID (Mostly due to student changing bag color) and numerical features with negative or incorrect values (age) were fixed.
2. Null test score and attendance were replaced with the corresponding duplicate response that contain the test score and attendance.
3. Take absolute value for negative numerical features.
4. Relabel data that has the same label meanings but are inconsistent (eg 'Yes' and 'yes').

### **2. Feature Engineering**
1. Some numerical features (eg. number of hours spent sleeping) were binned into categories.
2. Removal of features that did not influence test scores (eg. gender, mode of transport).

### **3. Modelling**
1. Output data is encoded using OneHotEncoder and scaled using StandardScaler and passed into 3 models: linear regression, random forest and XGBoost. The performance of the model is judged using root mean squared error.
