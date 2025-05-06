import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('Clientes-v2-tratados.csv')

print(df.head())

#codificação one_hot para 'estado_civil'
df = pd.concat([df, pd.get_dummies(df['estado_civil'], prefix='estado_civil')], axis=1)

print("\nDataFrame apos codificação one-hot para 'estado_civil':\n", df.head())

#codificação ordinais para 'nivel_educacao'
educacao_ordem = {'Ensino Fundamental': 1, 'Ensino Médio': 2, 'Ensino Superior': 3, 'Pós-graduação': 4}
df['nivel_educacional_ordinal'] = df['nivel_educacao'].map(educacao_ordem)

print("\nDataFrame após codificarção ordinal para 'nivel_educacao':\n", df.head())

# Transformar 'area_atuacao' em categoria codificadas usando o metodo .cat.codes
df['area_atuacao_cod'] = df['area_atuacao'].astype('category').cat.codes

print("\nDataFrame após transformar 'area_atuacao' em códigos numéricos:\n", df.head())

# LabelEncoder para 'estado'
# LabelEncoder converte cada valor único em números de 0 a n_classes-1
label_encoder = LabelEncoder()
df['estado_cod'] = label_encoder.fit_transform(df['estado'])

print("\nDataFrame após aplicar LbaelEncoder em 'estado':\n", df.head())
