import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Duomenų nuorodos
  # Oro kokybės duomenų failas
excel_file_ac = '*.csv'
  # Oro sąlygų duomenų failas
excel_files_aq = ['*.xlsx']
sheets = ['PM10', 'PM2.5', 'SO2', 'NO2', 'O3', 'CO']

# Duomenų nuskaitymas, agregavimas į dieninius ir jungimas į bendrą duomenų rinkinį
  # Oro kokybės duomenys
aq_data = pd.DataFrame()
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


  # Oro sąlygų duomenys
meteo_data = pd.read_csv(excel_file_ac, parse_dates=['Observation Time'])
meteo_data.columns = [col.strip() for col in meteo_data.columns]
for col in meteo_data.columns:
    if col not in ['Station', 'Observation Time']:
        meteo_data[col] = pd.to_numeric(meteo_data[col].replace(',', '.', regex=True), errors='coerce')
meteo_data.set_index(['Station', 'Observation Time'], inplace=True)
meteo_data = meteo_data.interpolate(method='linear')
meteo_daily = meteo_data.groupby(level='Station').resample('D', level='Observation Time').mean()


  # Stočių pavadinimų suvienodinimas
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

combined_data['Observation Time'] = pd.to_datetime(combined_data['Observation Time'])
combined_data.set_index('Observation Time', inplace=True)


# Duomenų nuskaitymo ir agregavimo pabaiga


# Duomenų normalizavimas
def normalize_data(data):
    scaler = MinMaxScaler()
    data_scaled = data.copy()
    for column in data.columns:
        data_scaled[column] = scaler.fit_transform(data[[column]])
    return data_scaled

# Oro kokybės ir hidrometeorologinių duomenų normalizavimas
combined_data.iloc[:, 3:] = normalize_data(combined_data.iloc[:, 3:])

stations = combined_data['Station'].unique()
pollutants = combined_data['Pollutant'].unique()


# Stočių pavadinimų keitimas
station_names = {
    'duksto-ams': 'Aukštaitija',
    'kauno-ams': 'Kaunas',
    'telsiu-ams': 'Mažeikiai',
    'vilniaus-ams': 'Vilnius'
}


# Oro kokybės stočių duomenų grafikų braižymas
def plot_air_quality(data, station_names):
    if not isinstance(station_names, dict):
        print("station_names must be a dictionary")
        return

    pollutants = data['Pollutant'].unique()
    for station, station_label in station_names.items():
        plt.figure(figsize=(10, 5))
        station_data = data[data['Station'] == station]
        for pollutant in pollutants:
            pollutant_data = station_data[station_data['Pollutant'] == pollutant]
            if not pollutant_data.empty:
                plt.plot(pollutant_data.index, pollutant_data['Value'], label=pollutant)
        plt.title(f'Oro kokybė: {station_label}')
        plt.xlabel('Data')
        plt.ylabel('Normalizuota reikšmė')
        plt.legend()
        plt.grid(True)
        plt.show()

plot_air_quality(combined_data, station_names)

# Oro sąlygų parametrų pervadinimas į lietuviškus
parameter_names_lt = {
    'Air Temperature': 'Oro temperatūra',
    'Precipitation': 'Krituliai',
    'Wind Speed': 'Vėjo greitis',
    'Sea Level Pressure': 'Slėgis jūros lygyje',
    'Cloud Cover': 'Debesuotumas',
    # Add more parameters as needed
}

# Hidrometerologinių stočių duomenų grafikų braižymas
def plot_meteorological_data(data, station_names):
    if not isinstance(station_names, dict):
        print("station_names must be a dictionary")
        return

    parameters = [col for col in data.columns if col not in ['Station', 'Pollutant', 'Value']]
    for station, station_label in station_names.items():
        station_data = data.loc[station]
        for parameter in parameters:
            lithuanian_param_name = parameter_names_lt.get(parameter, parameter)  # Default to English if no translation
            plt.figure(figsize=(10, 5))
            plt.plot(station_data.index, station_data[parameter], label=parameter)
            plt.title(f'{lithuanian_param_name}: {station_label}')
            plt.xlabel('Data')
            plt.ylabel('Normalizuota reikšmė')
            plt.legend()
            plt.grid(True)
            plt.show()

plot_meteorological_data(meteo_daily, station_names)
