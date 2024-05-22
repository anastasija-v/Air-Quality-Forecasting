# reikiamų bibliotekų ir paketų importavimas
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, Dropout



# duomenų failų nuorodos (path)
excel_file_ac = '~/api_data_fx.csv' # ac - air conditions (oro sąlygos)
excel_files_aq = ['~/Duomenys_2022_s2.xlsx', '~/Duomenys_2023_s2.xlsx'] # aq - air quality (oro kokybė)
sheets = ['PM10', 'PM2.5', 'SO2', 'NO2', 'O3', 'CO']


# tusčio duomenų rinkinio (DataFrame) sukūrimas, laikyti oro kokybės duomenims
aq_data = pd.DataFrame()
# oro kokybės duomenų įkėlimas, tučių reikšmių užpildymas interpoliuojant, agregavimas iš valandinių į dieninius ir struktūravimas
for file in excel_files_aq:
    xls = pd.ExcelFile(file)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, index_col=0, skipfooter=4, parse_dates=True)
        df_long = df.stack().reset_index()
        df_long.columns = ['Observation Time', 'Station', 'Value']
        df_long['Pollutant'] = sheet
        df_long = df_long[['Pollutant', 'Station', 'Observation Time', 'Value']]
        df_long['Value'] = pd.to_numeric(df_long['Value'].replace(',', '.', regex=True), errors='coerce')
        aq_data = pd.concat([aq_data, df_long], ignore_index=True)

aq_data.set_index(['Station', 'Pollutant', 'Observation Time'], inplace=True)
aq_data = aq_data.interpolate(method='linear')
daily_aq = aq_data.groupby(level=['Station', 'Pollutant']).resample('D', level='Observation Time').mean()


# hidrometorologinių duomenų įkėlimas, tučių reikšmių užpildymas interpoliuojant, agregavimas iš valandinių į dieninius ir struktūravimasmeteo_data = pd.read_csv(excel_file_ac, parse_dates=['Observation Time'])
meteo_data.columns = [col.strip() for col in meteo_data.columns]
for col in meteo_data.columns:
    if col not in ['Station', 'Observation Time']:
        meteo_data[col] = pd.to_numeric(meteo_data[col].replace(',', '.', regex=True), errors='coerce')
meteo_data.set_index(['Station', 'Observation Time'], inplace=True)
meteo_data = meteo_data.interpolate(method='linear')
meteo_daily = meteo_data.groupby(level='Station').resample('D', level='Observation Time').mean()


# stočių pavadinimų keitimas ir duomenų rinkinų jungimas
# stočių pavadinimai
station_map = {
    '0001=Vilnius, Senamiestis': 'vilniaus-ams',
    '0023=Mažeikiai': 'telsiu-ams',
    '0051=Aukštaitija': 'duksto-ams',
    '0041=Kaunas, Petrašiūnai': 'kauno-ams'
}

daily_aq.reset_index(inplace=True)
daily_aq['Station'] = daily_aq['Station'].map(station_map)
daily_aq.set_index(['Station', 'Pollutant', 'Observation Time'], inplace=True)

combined_data = pd.merge(daily_aq, meteo_daily, left_index=True, right_index=True, how='inner')
combined_data.reset_index(inplace=True)  # Necessary to align indices before dropna
combined_data.dropna(inplace=True)

# duomenų normalizavimas nuo 0 iki 1
scaler = MinMaxScaler(feature_range=(0, 1))
numeric_features = combined_data.select_dtypes(include=[np.number]).columns
combined_data[numeric_features] = scaler.fit_transform(combined_data[numeric_features])



# duomenų išskyrimo indeksų skaičiavimo funkcija
def calculate_split_indices(data, train_percent, validate_percent, test_percent):
    total_count = len(data)
    train_end = int(total_count * train_percent)
    validate_end = train_end + int(total_count * validate_percent)
    return train_end, validate_end


# stočių, teršalų ir prdiktorių apibrėžimas
stations = combined_data['Station'].unique()
pollutants = combined_data['Pollutant'].unique()
features = ['Air Temperature', 'Wind Speed', 'Precipitation', 'Sea Level Pressure', 'Cloud Cover']

# tusčių žodyno tipo duomenų rinkinių sukūrimas
models = {}
mse_results = {}

# būlio funcijos taikymas pasirenkant skaičiuoti ilgalaikės-trumpalaikės atminties (LSTM) ar vartinių rekurentinių vienetų (GRU) modelius 
use_lstm = False  # keisti norint alternuoti tarp LSTM ir GRU

# iteravimas per kiekvieną stotį
for station in stations:
    models[station] = {}
    mse_results[station] = {}
    station_data = combined_data[combined_data['Station'] == station]
    # iteravimas per kiekvieną teršalą
    for pollutant in pollutants:
        pollutant_data = station_data[station_data['Pollutant'] == pollutant].sort_index()
      # duomenų skyrimo indeksų skaičiavimas pagal pasirinktus procentus
        train_end, validate_end = calculate_split_indices(pollutant_data, 0.70, 0.15, 0.15)
      # duomenų išskyrimas į apmokymo, testavimo ir validavimo duomenų rinkinius
        train_data = pollutant_data.iloc[:train_end]
        validate_data = pollutant_data.iloc[train_end:validate_end]
        test_data = pollutant_data.iloc[validate_end:]
        # saugiklis tikrinti ar sukurti duomenų rinkiniai nėra tušti
        if train_data.empty or test_data.empty or validate_data.empty:
            print(f"Insufficient data for training, validating, or testing for {pollutant} at {station}. Skipping...")
            continue
        # reikšmių apibrėžimas
        X_train = train_data[features]
        y_train = train_data['Value']
        X_validate = validate_data[features]
        y_validate = validate_data['Value']
        X_test = test_data[features]
        y_test = test_data['Value']

        # pertvarkymas rekurentinių neuroninių tinklų įvesčiai
        X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_validate = X_validate.values.reshape((X_validate.shape[0], 1, X_validate.shape[1]))
        X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # alternavimas tarp LSTM ir GRU
        if use_lstm:
            rnn_layer = LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[2]))
            rnn_layer2 = LSTM(30, return_sequences=False)
        else:
            rnn_layer = GRU(50, return_sequences=True, input_shape=(1, X_train.shape[2]))
            rnn_layer2 = GRU(30, return_sequences=False)

      # modelio su pasirinktais parametrais sukūrimas
        model = Sequential([
            rnn_layer,
            Dropout(0.2),
            rnn_layer2,
            Dense(20, activation='relu'),
            Dense(1)
        ])

      # optimizavimas Adam metodu
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=300, batch_size=10, validation_data=(X_validate, y_validate))

        # prognozavimas ir vertinimas
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_results[station][pollutant] = mse
        
        # indekso į laiko tipo objektą (datetime) keitimas
        test_data['Observation Time'] = pd.to_datetime(test_data['Observation Time'])
        validate_data['Observation Time'] = pd.to_datetime(validate_data['Observation Time'])
        test_data.set_index('Observation Time', inplace=True)
        validate_data.set_index('Observation Time', inplace=True)



        # rezultatų braižymas
        plt.figure(figsize=(10, 6))
        plt.plot(test_data.index, y_test, label='Tikros reikšmės', color='blue')
        plt.plot(test_data.index, y_pred.flatten(), label='Prognozuojamos reikšmės', color='orange')
        if station == 'kauno-ams':
            plt.title(f'Tikri ir prognozuojami {pollutant} lygiai: Kaunas')
        elif station == 'duksto-ams':
            plt.title(f'Tikri ir prognozuojami {pollutant} lygiai: Aukštaitija')
        elif station == 'vilniaus-ams':
            plt.title(f'Tikri ir prognozuojami {pollutant} lygiai: Vilnius')
        else:
            plt.title(f'Tikri ir prognozuojami {pollutant} lygiai: Mažeikiai')

        plt.xlabel('Data')
        plt.ylabel(f'{pollutant} lygis')
        plt.legend()
        plt.show()
        # Toggle the model type for the next iteration
        use_lstm = not use_lstm

# Vidutinės kvadratinės paklaidos rezultatų spausdinimas
for station in mse_results:
    for pollutant, mse in mse_results[station].items():
        print(f"{station} - {pollutant} MSE: {mse}")
