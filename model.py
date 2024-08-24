import pandas as pd 
 
import pycaret 
pycaret.__version__ 
 
data = pd.read_csv('03_transaction_records.csv') 
 
data.drop_duplicates(inplace=True) 
 
data['DIV_NAME'] = data['DIV_NAME'].apply(lambda x: x.upper()) 
data['CAT_DESC'] = data['CAT_DESC'].apply(lambda x: x.upper()) 
data['AMT'] = data['AMT'].abs() 
data['MERCHANT'] = data['MERCHANT'].apply(lambda x: x.upper()) 
 
 
from pycaret.anomaly import * 
s = setup(data,  
          ignore_features = ['DEPT_NAME', 'FISCAL_YR', 'FISCAL_MTH'], 
          session_id = 123) 
 
 
knn=create_model('knn') 
# plot_model(knn) 

save_model(knn, 'anomaly_detection')