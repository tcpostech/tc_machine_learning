import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

st.set_page_config(page_title="Predição de apartamentos em São Paulo", layout="wide")
st.title('Predição de apartamentos em São Paulo')

# Link do arquivo: https://www.kaggle.com/datasets/kaggleshashankk/house-price-data-of-sao-paulo
st.subheader('Lista de apartamentos antes do tratamento de dados')
df = pd.read_excel('data/Sao paulo.xlsx', engine='openpyxl')
df = df.rename(columns={'Street': 'rua', 'City': 'bairro', 'propertycard__detailvalue': 'area', 'price': 'preco'})
df.dropna(inplace=True)
st.write(df)

st.subheader('Lista de apartamentos após o tratamento de dados')
df['rua'] = df['rua'].astype(str)
df['bairro'] = df['bairro'].astype(str)
df['area'] = df['area'].str.replace(r'-\d+', '', regex=True).astype(int)
df['quartos'] = df['quartos'].str.replace(r'-\d+', '', regex=True).astype(int)
df['banheiros'] = df['banheiros'].str.replace(r'-\d+', '', regex=True).astype(int)
df['vagas'] = df['vagas'].str.replace(r'-\d+', '', regex=True).astype(int)
df['preco'] = df['preco'].str.replace(r'\D', '', regex=True).astype(int)
df['bairro'] = df.apply(lambda x: x['bairro'].replace('SP', x['rua']), axis=1)
st.write(df)

# Existem alguns preços e áreas como outliers que precisam ser removidos
df.drop([9628])
df = df[df.preco > 250000]
df = df[df.preco < 37000000]
df = df[df.area < 1050]

st.subheader('Lista de apartamentos por bairro')
bairros = df.bairro.value_counts()
st.write(bairros)

# Análise exploratória
st.subheader('Estatísticas descritivas')
st.write(df.describe())

st.subheader('Correlação entre variáveis numéricas')
corr = df[['area', 'quartos', 'banheiros', 'vagas', 'preco']].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Mapa de Correlação')
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader('Gráficos')
# Gráfico 01
fig_01 = px.scatter(df, x="area", y="preco", labels={"area": "Área (m²)", "preco": "Preço (R$)"},
                    color="bairro", title="Dispersão de Área vs Preço por Bairro")
fig_01.update_layout(legend_title_text='Bairro', legend=dict(font=dict(size=10)))
st.plotly_chart(fig_01, use_container_width=True)

# Gráfico 02
fig_02 = px.scatter(df, x='area', y='preco', labels={"area": "Área (m²)", "preco": "Preço (R$)"}, trendline='lowess',
                    trendline_color_override='red', title="Dispersão de Área vs Preço com Tendência LOWESS")
st.plotly_chart(fig_02, use_container_width=True)

# Gráfico 03
fig_03 = px.box(df, x='quartos', y='preco', labels={"quartos": "Quartos", "preco": "Preço (R$)"},
                title="Boxplot de Preço por Quartos")
st.plotly_chart(fig_03, use_container_width=True)
st.divider()

st.subheader('Exibindo os dados de treinamento')
# Treinando o modelo e exibindo os dados
df_copy = df
df_bairros = pd.get_dummies(df_copy['bairro'], dtype=int)
df_copy = df_copy.drop(['bairro'], axis=1)
df_copy = df_copy.join(df_bairros)

x = df_copy.drop(['rua', 'preco', 'banheiros'], axis=1)
y = df_copy['preco']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=100)
train_data = x_train.join(y_train)
train_data.corr(numeric_only=True)

# Utilizando Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

# Utilizando Random Forest
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

df_models = pd.concat([lr_results, rf_results], axis=0)
df_models.reset_index(drop=True, inplace=True)
st.write(df_models)
# st.divider()
#
# forest = RandomForestRegressor()
# forest.fit(x_train, y_train)
# st.markdown(f"- Random forest Score: {forest.score(x_test, y_test)}")
#
# forest = RandomForestRegressor()
#
# param_grid = {
#     'n_estimators': [3, 10, 30],
#     'max_features': [2, 4, 6, 8],
#     'min_samples_split': [2, 4, 6, 8],
#     'max_depth': [None, 4, 8]
# }
#
# grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
# grid_search.fit(x_train, y_train)
#
# st.write(f"- Best estimators : {grid_search.best_estimator_}")
# st.markdown(f"- Grid Search Score: {grid_search.best_estimator_.score(x_test, y_test)}")
