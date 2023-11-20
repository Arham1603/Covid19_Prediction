# Time Series Project (Covid 19 New Case Prediction)

## Description

This project aims to build a deep learning model utilizing LSTM neural networks for predicting new COVID-19 cases (cases_new) in Malaysia. The prediction is based on the historical data of the past 30 days, considering the following features: cases_new, cases_import, cases_recovered, and cases_active. Two scenarios are considered:

Single Step Window:
Input width: 30
Output width: 30
Offset: 1
Multi-Step Window:
Input width: 30
Output width: 30
Offset: 30
The goal is to train the LSTM model to effectively capture temporal dependencies and patterns in the dataset, enabling accurate predictions of new COVID-19 cases. The model's performance will be evaluated based on its ability to handle both single-step and multi-step forecasting scenarios.

## Model Architecture

### Single Step:
![single_step](https://github.com/Arham1603/Covid19_Prediction/assets/150897618/a829d66d-6066-4bec-b4eb-78393522aa30)

### Multi Step:
![multi_step](https://github.com/Arham1603/Covid19_Prediction/assets/150897618/25c1c2ef-3ba6-4de5-aa66-d81069f5a3ad)


## Prediction Plot Example

### Single Step:
![single_graph](https://github.com/Arham1603/Covid19_Prediction/assets/150897618/20d448cf-63df-45ee-a9b8-b08eda7e82f1)

### Multi Step:
![multi_graph](https://github.com/Arham1603/Covid19_Prediction/assets/150897618/8d85f629-ef4b-417b-9b75-ac97c3957b5a)


## Data Source:
GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.
