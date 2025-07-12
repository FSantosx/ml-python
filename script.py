import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet

data = yf.download('JNJ', start='2020-01-01', end='2025-06-29', progress=False, auto_adjust=True)
data = data.reset_index()

training_data = data[data['Date'] < '2025-01-01']
test_data = data[data['Date'] >= '2025-01-01']

prophet_training_data = training_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_training_data.columns = [col[0] if isinstance(col, tuple) else col for col in prophet_training_data.columns]
prophet_training_data['y'] = pd.to_numeric(prophet_training_data['y'])

model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
)

model.add_country_holidays(country_name='US')
model.fit(prophet_training_data)


future = model.make_future_dataframe(periods=150)
prediction = model.predict(future)

test_data_copy = test_data.copy()
test_data_copy.columns = [col[0] for col in test_data.columns.values]

merged = pd.merge(
    test_data_copy[['Date', 'Close']],
    prediction[['ds', 'yhat']],
    left_on='Date', right_on='ds',
    how='inner'
)

rmse = np.sqrt(np.mean((merged['Close'] - merged['yhat'])**2))
print(f'RMSE: {rmse:.2f}')

plt.figure(figsize=(14, 7))
plt.plot(training_data['Date'], training_data['Close'], label='Dados de Treino', color='blue')
plt.plot(test_data['Date'], test_data['Close'], label='Dados de Reais (teste)', color='green')
plt.plot(prediction['ds'], prediction['yhat'], label='Previsão', color='orange', linestyle='--')

plt.axvline(training_data['Date'].max(), color='red', linestyle='--', label='Início da Previsão')
plt.xlabel('Data')
plt.ylabel('Preço de faturamento')
plt.title('Previsão de preço de Fechamento vs Dados Reais')
plt.legend()
plt.show()