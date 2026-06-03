import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/house_data.csv')
x = data[['area']]
y = data['price']
model = LinearRegression()
model.fit(x, y)
joblib.dump(model,'model.pkl')
print("Model trained, saved successfully added to github")