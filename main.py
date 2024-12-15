import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline


"""Settings of terminal setup"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


directory = r"Urban Air Quality and Health Impact Dataset.csv"
df = pd.read_csv(directory)

"""Deleting cols with blank lines"""
df = df.drop(["preciptype", "stations", "Condition_Code", "snowdepth", "sunrise", "sunset", "datetime", "snow"], axis=1)

"""Defining different groups of columns that will be processed using ColumnTransformer"""

scal_cols = ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike", "dew", "humidity",
                     "precip", "precipprob", "windgust", "windspeed", "winddir", "pressure", "cloudcover",
                     "visibility", "solarradiation", "solarenergy", "uvindex", "severerisk", "moonphase", "Temp_Range",
                     "Heat_Index", "Severity_Score"]

ordinal_cols = ["conditions", "description", "icon", "source", "Day_of_Week", "Is_Weekend", "Season"]
hot_cols = ["City"]

"""Preprocessing and scaling using ColumnTransformer"""
ordinal_encoder = OrdinalEncoder()
hot_encoder = OneHotEncoder()
scaler_pipline = make_pipeline(
    MinMaxScaler()
)

preprocessing_pipline = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_encoder, ordinal_cols),
        ('onehot', hot_encoder, hot_cols),
        ('scaling', scaler_pipline, scal_cols)
    ]
)

"""Making new DataFrame"""
data_preprocessed = pd.DataFrame(
    preprocessing_pipline.fit_transform(df),
    columns=ordinal_cols + list(preprocessing_pipline.named_transformers_['onehot'].get_feature_names_out(hot_cols)) + scal_cols,
    index=df.index
)
df_transformed = pd.DataFrame(preprocessing_pipline.fit_transform(df),
                              columns=ordinal_cols + list(preprocessing_pipline.named_transformers_['onehot'].get_feature_names_out(hot_cols)) + scal_cols,
                              index=df.index)

# Usuń stare kolumny dopiero teraz i dołącz nowe
df = df.drop(columns=scal_cols + ordinal_cols + hot_cols).join(df_transformed)

print(df.head())

model = LinearRegression()
"""Train test split was used to split dataset into training and testing sets"""
# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Health_Risk_Score']), df['Health_Risk_Score'],
                                                    test_size=0.2, random_state=42)

"""Splitting the training set into training and validation sets"""
# Zastosowanie walidacji krzyżowej, ponieważ pozwala ona na zminimalizowanie wpływu losowego podziału danych
# na jakość modelu oraz wyniki są uśredniane, co daje bardziej stabilną ocenę jakości modelu
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja wartości dla zbioru walidacyjnego
y_predict_val = model.predict(X_val)

mse = mean_squared_error(y_val, y_predict_val)
mae = mean_absolute_error(y_val, y_predict_val)

print(f"Mean squared error: {mse}")
print(f"Mean absolute error: {mae}")
