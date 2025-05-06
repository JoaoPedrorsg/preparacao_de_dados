import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('Clientes-v2-tratados.csv')

print(df.head())

df = df[['idade', 'salario']]

#normalização - MinMaxScaler
scaler = MinMaxScaler()
df['idadeMinMaxScaler'] = scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler'] = scaler.fit_transform(df[['salario']])

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
df['idadeMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['salario']])

#Padronização - StanderdScaler
scaler = StandardScaler()
df['idadeStanderdScaler'] = scaler.fit_transform(df[['idade']])
df['salararioStanderdScaler'] = scaler.fit_transform(df[['salario']])

#padronização - RobustScaler
scaler = RobustScaler()
df['idadeRobustScaler'] = scaler.fit_transform(df[['idade']])
df['salarioRobustScaler'] = scaler.fit_transform(df[['salario']])

print(df.head(15))

#MinMaxScaler
print("MinmaxScaler (de 0 a 1:")
print("Idade - min: {:.4f} Max: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['idadeMinMaxScaler'].min(),
    df['idadeMinMaxScaler'].max(),
    df['idadeMinMaxScaler'].mean(),
    df['idadeMinMaxScaler'].std()
))
print("\nSalario - min: {:.4f} Max: {:.4f} mean: {:.4f} Std: {:.4f}".format(
    df['salarioMinMaxScaler'].min(),
    df['salarioMinMaxScaler'].max(),
    df['salarioMinMaxScaler'].mean(),
    df['salarioMinMaxScaler'].std()
))

print("\nMinMaxScaler (de -1 a 1)")
print("Idade - min: {:.4f} Max: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['idadeMinMaxScaler_mm'].min(),
    df['idadeMinMaxScaler_mm'].max(),
    df['idadeMinMaxScaler_mm'].mean(),
    df['idadeMinMaxScaler_mm'].std()
))
print("salario - Min: {:.4f} Max: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['salarioMinMaxScaler_mm'].min(),
    df['salarioMinMaxScaler_mm'].max(),
    df['salarioMinMaxScaler_mm'].mean(),
    df['salarioMinMaxScaler_mm'].std()
))
#StanderdScaler
print("\nStanderdScaler (Ajuste a média a 0 e desvio padrão 1):")
print("Idade - min: {:.4f} Max: {:.4f} Mean: {:.18f} Std: {:.4f}".format(
    df['idadeStanderdScaler'].min(),
    df['idadeStanderdScaler'].max(),
    df['idadeStanderdScaler'].mean(),
    df['idadeStanderdScaler'].std()
))
print("Salario - min: {:.4f} Max: {:.4f} Mean: {:.18f} Std: {:.4}".format(
    df['salararioStanderdScaler'].min(),
    df['salararioStanderdScaler'].max(),
    df['salararioStanderdScaler'].mean(),
    df['salararioStanderdScaler'].std()
))

#RobustScaler
print("\nRobustScaler (Ajuste a média e IQR)")
print("Idade - Min: {:.4f} Max: {:.4} Mean: {:.4f} Std: {:.4f}".format(
    df['idadeRobustScaler'].min(),
    df['idadeRobustScaler'].max(),
    df['idadeRobustScaler'].mean(),
    df['idadeRobustScaler'].std()
))
print("Salario - Min: {:.4f} Max: {:.4f} Mean: {:.4} Std: {:.4f}".format(
    df['salarioRobustScaler'].min(),
    df['salarioRobustScaler'].max(),
    df['salarioRobustScaler'].mean(),
    df['salarioRobustScaler'].std()
))
