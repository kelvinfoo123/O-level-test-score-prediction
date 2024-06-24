# O-level-test-score-prediction
### **1. Preprocessing**
1. Null features and numerical features with negative values are processed and fixed as their values are incorrect
2. Null features are replaced with the median values while numerical features take an abs() value
3. Relabel data that has the same label meanings but are inconsistent (eg 'Yes' and 'yes')
