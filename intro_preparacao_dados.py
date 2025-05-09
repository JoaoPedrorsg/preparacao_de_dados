import pandas as pd

df = pd.read_csv(r'C:\Users\joaoz\OneDrive\Documentos\clientes-v2.csv')

print(df.head().to_string())
print(df.tail().to_string())

print('Verificação inicial:\n')
print(df.info())

print('Analise de dados nulos:\n', df.isnull().sum())
print('% de dados nulos:\n', df.isnull().mean() * 100)
df.dropna(inplace=True)
print('Confirmar remoção de dados nulos:\n', df.isnull().sum().sum())

print('Analise de dados duplicadps:\n', df.duplicated().sum())

print('Analise de dados unicos:\n', df.nunique())

print('Estatísticas dos dados:\n', df.describe())

df = df[['idade', 'data', 'estado', 'salario', 'nivel_educacao', 'numero_filhos', 'estado_civil', 'area_atuacao']]
print(df.head().to_string())

df.to_csv('Clientes-v2-tratados.csv', index=False)
