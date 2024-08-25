# python file for building the model as the pkl file is too large to push into GitHub
# run this file to the get knn model pickle file

import pandas as pd 
 
import pycaret 
pycaret.__version__ 

# reading data in 
data = pd.read_csv('03_transaction_records.csv') 
 
# performing data cleaning that cannot be done in the set up
data.drop_duplicates(inplace=True) 
data['DIV_NAME'] = data['DIV_NAME'].apply(lambda x: x.upper()) 
data['CAT_DESC'] = data['CAT_DESC'].apply(lambda x: x.upper()) 
data['AMT'] = data['AMT'].abs() 
data['MERCHANT'] = data['MERCHANT'].apply(lambda x: x.upper()) 
 
# initialize the set up 
from pycaret.anomaly import * 
s = setup(data,  
          ignore_features = ['DEPT_NAME', 'FISCAL_YR', 'FISCAL_MTH'], 
          session_id = 123) 
 
# build the model 
knn=create_model('knn') 
# plot_model(knn) 

# save to pkl file
save_model(knn, 'anomaly_detection')