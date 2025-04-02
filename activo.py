'''
 Streamlit del proyecto 
 
 '''
import streamlit as st #para la página
import pandas as pd #para dataframes
import numpy as np
import yfinance as yf # extraccion de precios de activos financieros
import matplotlib.pyplot as plt
from scipy.stats import* # funciones estadísticas

st.title("Proyecto 1")
st.header("Métodos Cuantitativos en Finanzas")


#Funciones
def obtener_datos(activo): 
    
    # Descargar los datos
    df = yf.download(activo, start = "2010-01-01", progress=False )['Close']
    return df

def calcular_rendimientos(df):
     #return df.pct_change().dropna()
     return np.log(df).diff().dropna()


#Activo
activo_escogido = ['BTC-USD']
 
 #Carga de página
with st.spinner("Descargando datos..."):
     df_precios = obtener_datos(activo_escogido)
     df_rendimientos = calcular_rendimientos(df_precios)
 
if activo_escogido:    
     #Métricas del activo
     st.subheader(f"Métricas de Rendimiento: {activo_escogido[0]}")
     rendimiento_medio = df_rendimientos[activo_escogido[0]].mean() #Columna de los datos
     Kurtosis = kurtosis(df_rendimientos[activo_escogido[0]])
     Skew = skew(df_rendimientos[activo_escogido[0]])

     #Columnas
     col1, col2, col3= st.columns(3)
     col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
     col2.metric("Kurtosis", f"{Kurtosis:.4}")
     col3.metric("Skew", f"{Skew:.2}")
     

     # Gráfico de rendimientos diarios
     st.subheader(f"Gráfico de Rendimientos: {activo_escogido[0]}")
     fig, ax = plt.subplots(figsize=(13, 5))
     ax.plot(df_rendimientos.index, df_rendimientos[activo_escogido], label=activo_escogido)
     ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
     ax.legend()
     ax.set_title(f"Rendimientos de {activo_escogido[0]}")
     ax.set_xlabel("Fecha")
     ax.set_ylabel("Rendimiento Diario")
     st.pyplot(fig)
     
     # Histograma de rendimientos
     st.subheader("Distribución de Rendimientos")
     fig, ax = plt.subplots(figsize=(10, 5))
     ax.hist(df_rendimientos[activo_escogido], bins=30, alpha=0.7, color='blue', edgecolor='black')
     ax.axvline(rendimiento_medio, color='red', linestyle='dashed', linewidth=2, label=f"Promedio: {rendimiento_medio:.4%}")
     ax.legend()
     ax.set_title("Histograma de Rendimientos")
     ax.set_xlabel("Rendimiento Diario")
     ax.set_ylabel("Frecuencia")
     st.pyplot(fig)


     st.subheader('Test de Normalidad (Shapiro-Wilk)')

     stat , p = shapiro(df_rendimientos[activo_escogido[0]])
     st.write(f'Shapirop-Wilk Test : {stat:.4}' )
     st.write(f'P_value : {p:.6}')


     
     #VaR paramétrico
     
     st.header("VaR (Asuminedo una distribucion Normal)")


     import streamlit as st

     # Diccionario correcto sin lista
     porcentaje_confianza = {'95%': 0.95, '97.5%': 0.975, '99%': 0.99}

     var_seleccionado = st.selectbox("Selecciona un porcentaje de confianza", list(porcentaje_confianza.keys()))

    # Convertir el porcentaje seleccionado a su valor numérico
     valor_confianza = porcentaje_confianza[var_seleccionado]

     porcentaje = 1 - valor_confianza
 
if var_seleccionado:
     
     st.subheader("Aproximación paramétrica")
     promedio = np.mean(df_rendimientos[activo_escogido[0]])
     stdev = np.std(df_rendimientos[activo_escogido[0]])


     VaR = norm.ppf(porcentaje,promedio,stdev)

     col4, = st.columns(1)
     col4.metric(f"VaR con {var_seleccionado} de confianza", f"{VaR:.4}")
     

     st.subheader("Aproximación Histórica")
     # Historical VaR
     hVaR = (df_rendimientos[activo_escogido[0]].quantile(porcentaje))

     col5, = st.columns(1)
     col5.metric(f"hVaR con {var_seleccionado} de confianza", f"{hVaR:.4}")
     

     st.subheader("Monte Carlo")
    # Monte Carlo
    # Number of simulations
     n_sims = 100000

      # Simulate returns and sort
     sim_returns = np.random.normal(promedio, stdev, n_sims)

     MCVaR = np.percentile(sim_returns, porcentaje*100)

     col10, = st.columns(1)
     col10.metric(f"MCVaR con {var_seleccionado} de confianza", f"{MCVaR:.4}")

     st.subheader("CVaR (ES)")
     CVaR= (df_rendimientos[activo_escogido[0]][df_rendimientos[activo_escogido[0]] <= hVaR].mean())
     
     col13, = st.columns(1)
     col13.metric(f"CVaR con {var_seleccionado} de confianza", f"{CVaR:.4}")


#Gráfica
     fig_2, ax_2 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR_95 and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_2.axvline(x=VaR, color='black', linestyle='--', label= f'VaR {var_seleccionado} (Paramétrico)')
     ax_2.axvline(x=MCVaR, color='grey', linestyle='--', label=f'VaR {var_seleccionado} (Monte Carlo)')
     ax_2.axvline(x=hVaR, color='green', linestyle='--', label=f'VaR {var_seleccionado}(Aprox.Histórico)')
     ax_2.axvline(x=CVaR, color='purple', linestyle='-.', label= f'CVaR {var_seleccionado}')

# Add a legend and labels to make the chart more informative
     ax_2.set_title(f'Histograma de los Retornos con VaR y CVaR al {var_seleccionado}')
     ax_2.set_xlabel('Retornos')
     ax_2.set_ylabel('Frequencia')
     ax_2.legend()
     st.pyplot(fig_2)

       # Create a DataFrame with various VaR and CVaR calculations
     out = pd.DataFrame({'VaR (Normal)': [VaR * 100],
                    'VaR (Historical)': [hVaR * 100],
                    'VaR (Monte Carlo)': [MCVaR * 100],
                    'CVaR': [CVaR * 100]},
                   index=[f'{var_seleccionado} Confidence'])

# Display the DataFrame
     out


     st.header("VaR (Asuminedo una distribucion t de Student)")

     df_grados_de_libertad = len(df_rendimientos)-1

     st.subheader("VaR Paramétrico")

     t_cuantil = t.ppf(porcentaje, df_grados_de_libertad )
     VaR_t = promedio + t_cuantil * stdev
     

     col16,= st.columns(1)
     col16.metric(f"VaR con {var_seleccionado} de confianza (t-Student)", f"{VaR_t:.4}")

     
     
     st.subheader("Aproximación Histórica")
     # Historical VaR
     hVaR = (df_rendimientos[activo_escogido[0]].quantile(porcentaje))



     col6, = st.columns(1)
     col6.metric(f"hVaR con {var_seleccionado} de confianza", f"{hVaR:.4}")




     st.subheader("Monte Carlo")
    # Monte Carlo
    # Number of simulations
     n_sims = 100000

      # Simulate returns and sort
     sim_returns = promedio + stdev * np.random.standard_t(df_grados_de_libertad, size=n_sims)

     MCVaR_t = np.percentile(sim_returns, porcentaje*100)
     

     col19,  = st.columns(1)
     col19.metric(f"MCVaR con {var_seleccionado} de confianza", f"{MCVaR_t:.4}")

     st.subheader("CVaR (ES)")
     CVaR= (df_rendimientos[activo_escogido[0]][df_rendimientos[activo_escogido[0]] <= hVaR].mean())
     
     col7, = st.columns(1)
     col7.metric(f"CVaR con {var_seleccionado} de confianza", f"{CVaR:.4}")
     
    #Gráfica
     fig_3, ax_3 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_3.axvline(x=VaR_t, color='black', linestyle='--', label=f'VaR (t de student) {var_seleccionado} (Paramétrico)')
     ax_3.axvline(x=MCVaR_t, color='grey', linestyle='--', label=f'VaR {var_seleccionado} (Monte Carlo)')
     ax_3.axvline(x=hVaR, color='green', linestyle='--', label=f'VaR {var_seleccionado} (Aprox. Histórico)')
     ax_3.axvline(x=CVaR, color='purple', linestyle='-.', label=f'CVaR {var_seleccionado}')

# Add a legend and labels to make the chart more informative
     ax_3.set_title(f'Histograma de los Retornos con VaR y CVaR al {var_seleccionado}')
     ax_3.set_xlabel('Retornos')
     ax_3.set_ylabel('Frequencia')
     ax_3.legend()
     st.pyplot(fig_3)


#Rolling Window

     rolling_mean = df_rendimientos[activo_escogido[0]].rolling(window= 252).mean()


     rolling_std = df_rendimientos[activo_escogido[0]].rolling(window= 252).std()

     VaR95_rolling = norm.ppf(1-0.95, rolling_mean, rolling_std)


     VaR95_rolling_percent = (VaR95_rolling * 100).round(4)

     VaR_rolling_df = pd.DataFrame({'Date': df_rendimientos[activo_escogido[0]].index, '95% VaR Rolling': VaR95_rolling_percent.squeeze()})
     VaR_rolling_df.set_index('Date', inplace=True)

 
     VaR_rolling_df.tail()

     plt.figure(figsize=(14, 7))

# Plot daily returns
# Assuming returns are expressed as fractions, multiply by 100 to convert to percentage
     plt.plot(df_rendimientos[activo_escogido[0]].index, df_rendimientos[activo_escogido[0]] * 100, label='Daily Returns (%)', color='blue', alpha=0.5)

# Plot the 95% Rolling VaR
     plt.plot(VaR_rolling_df.index, VaR_rolling_df['95% VaR Rolling'], label='95% Rolling VaR', color='red')

# Add a title and axis labels
     plt.title('Daily Returns and 95% Rolling VaR')
     plt.xlabel('Date')
     plt.ylabel('Values (%)')

# Add a legend
     plt.legend()

# Show the plot
     plt.tight_layout()
     plt.show()
