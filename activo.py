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
     return df.pct_change().dropna()
     #return np.log(df).diff().dropna()


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
     ax.plot(df_rendimientos.index, df_rendimientos[activo_escogido], label='Rendimientos Diarios (%)')
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


     # Diccionario correcto sin lista
     porcentaje_confianza = {'95%': 0.95, '97.5%': 0.975, '99%': 0.99}

     var_seleccionado = st.selectbox("Selecciona un porcentaje de confianza", list(porcentaje_confianza.keys()))

    # Convertir el porcentaje seleccionado a su valor numérico
     valor_confianza = porcentaje_confianza[var_seleccionado]

     porcentaje = 1 - valor_confianza
 
if var_seleccionado:
     
     #st.subheader("Aproximación paramétrica")
     promedio = np.mean(df_rendimientos[activo_escogido[0]])
     stdev = np.std(df_rendimientos[activo_escogido[0]])


     VaR = norm.ppf(porcentaje,promedio,stdev)

     #col4, = st.columns(1)
     #col4.metric(f"VaR con {var_seleccionado} de confianza", f"{VaR:.4}")
     

     #st.subheader("Aproximación Histórica")
     # Historical VaR
     hVaR = (df_rendimientos[activo_escogido[0]].quantile(porcentaje))

     #col5, = st.columns(1)
     #col5.metric(f"hVaR con {var_seleccionado} de confianza", f"{hVaR:.4}")
     

     #st.subheader("Monte Carlo")
    # Monte Carlo
    # Number of simulations
     n_sims = 100000

      # Simulate returns and sort
     sim_returns = np.random.normal(promedio, stdev, n_sims)

     MCVaR = np.percentile(sim_returns, porcentaje*100)

     #col10, = st.columns(1)
     #col10.metric(f"MCVaR con {var_seleccionado} de confianza", f"{MCVaR:.4}")

     #st.subheader("CVaR (ES)")
     CVaR= (df_rendimientos[activo_escogido[0]][df_rendimientos[activo_escogido[0]] <= hVaR].mean())
     
     #col13, = st.columns(1)
     #col13.metric(f"CVaR con {var_seleccionado} de confianza", f"{CVaR:.4}")
     


     col4, col5, col10, col13 = st.columns(4)
     col4.metric(f"VaR con {var_seleccionado} de confianza", f"{VaR:.4}")
     col5.metric(f"hVaR con {var_seleccionado} de confianza", f"{hVaR:.4}")
     col10.metric(f"MCVaR con {var_seleccionado} de confianza", f"{MCVaR:.4}")
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

     #st.subheader("VaR Paramétrico")

     t_cuantil = t.ppf(porcentaje, df_grados_de_libertad )
     VaR_t = promedio + t_cuantil * stdev
     

     #col16,= st.columns(1)
     #col16.metric(f"VaR con {var_seleccionado} de confianza (t-Student)", f"{VaR_t:.4}")

     
     
     #st.subheader("Aproximación Histórica")
     # Historical VaR
     #hVaR = (df_rendimientos[activo_escogido[0]].quantile(porcentaje))



     st.subheader("Monte Carlo")
    # Monte Carlo
    # Number of simulations
     n_sims = 100000

      # Simulate returns and sort
     sim_returns = promedio + stdev * np.random.standard_t(df_grados_de_libertad, size=n_sims)

     MCVaR_t = np.percentile(sim_returns, porcentaje*100)
     

     #col19,  = st.columns(1)
     #col19.metric(f"MCVaR con {var_seleccionado} de confianza", f"{MCVaR_t:.4}")

     #st.subheader("CVaR (ES)")
     #CVaR= (df_rendimientos[activo_escogido[0]][df_rendimientos[activo_escogido[0]] <= hVaR].mean())
     
     

     col16, col5, col19, col13 =  st.columns(4)
     col16.metric(f"VaR con {var_seleccionado} de confianza", f"{VaR_t:.4}")
     col5.metric(f"hVaR con {var_seleccionado} de confianza", f"{hVaR:.4}")
     col19.metric(f"MCVaR con {var_seleccionado} de confianza", f"{MCVaR_t:.4}")
     col13.metric(f"CVaR con {var_seleccionado} de confianza", f"{CVaR:.4}")




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


     #### Nuevo





     
     #Calculo de medias y desviación estandar para Rolling Window de 252 días
     rolling_mean = df_rendimientos[activo_escogido[0]].rolling(window= 252).mean()
     rolling_std = df_rendimientos[activo_escogido[0]].rolling(window= 252).std()

     #Cálculos de VaR al 95% de confianza, creación de dataframe para poder graficarlo
     VaR_rolling = norm.ppf(porcentaje, rolling_mean, rolling_std)
     VaR_rolling_percent = (VaR_rolling * 100).round(4)
     VaR_rolling_df = pd.DataFrame({'Date': df_rendimientos[activo_escogido[0]].index, f'{var_seleccionado} VaR Rolling': VaR_rolling_percent.squeeze()})
     VaR_rolling_df.set_index('Date', inplace=True)
     
     
     #Cálculo de hVaR al 95% de confianza, creación de dataframe para graficarlo y poder sacar el ES
     hVaR_rolling = (df_rendimientos[activo_escogido[0]].rolling(window = 252).quantile(porcentaje))
     hVaR_rolling_percent = (hVaR_rolling * 100).round(4)
     hVaR_rolling_df = pd.DataFrame({'Date': df_rendimientos[activo_escogido[0]].index, f'{var_seleccionado} hVaR Rolling': hVaR_rolling_percent.squeeze()})
     hVaR_rolling_df.set_index('Date', inplace=True)
     
     
    # Cálculo del VaR al 95% (rolling)
     z = norm.ppf(valor_confianza)  # Percentil 95% de la normal estándar
     VaR_rolling = rolling_mean - z * rolling_std  # VaR al 95%

# Cálculo del CVaR al 95% (Expected Shortfall)
     CVaR_rolling = rolling_mean - (norm.pdf(z) / (porcentaje)) * rolling_std

# Convertir a porcentaje
     CVaR_rolling_percent = (CVaR_rolling * 100).round(4)

# Crear DataFrame para graficar
     CVaR_rolling_df = pd.DataFrame({
         'Date': df_rendimientos.index,  # Asegurar que se usa el índice correcto
         f'{var_seleccionado} CVaR Rolling': CVaR_rolling_percent.squeeze()})
     

# Establecer índice de fecha
     CVaR_rolling_df.set_index('Date', inplace=True)
     
     #Cálculo del hCVaR al 95% de confianza, creación de dataframe para graficarlo
     hCVaR_rolling = df_rendimientos[activo_escogido[0]].rolling(window=252).apply(lambda x: x[x <= hVaR_rolling.loc[x.index[-1]]].mean(), raw=False)
     hCVaR_rolling_percent = (hCVaR_rolling * 100).round(4)
     hCVaR_rolling_df = pd.DataFrame({'Date': df_rendimientos[activo_escogido[0]].index, f'{var_seleccionado} hCVaR Rolling': hCVaR_rolling_percent.squeeze()})
     hCVaR_rolling_df.set_index('Date', inplace=True)

     # Gráfico de rendimientos diarios y Rolling window VaR, hVaR, CVaR al 95% y 99% de confianza 
     st.subheader(f"Gráfico de Rendimientos rolling Window: {activo_escogido[0]}")
     fig_4, ax_4 = plt.subplots(figsize=(14, 7))
     ax_4.plot(df_rendimientos.index, df_rendimientos[activo_escogido] * 100, label=activo_escogido, color = 'blue', alpha = 0.5)
     ax_4.plot(VaR_rolling_df.index, VaR_rolling_df[f'{var_seleccionado} VaR Rolling'], label=f'{var_seleccionado} Rolling VaR', color='red')
     ax_4.plot(hVaR_rolling_df.index, hVaR_rolling_df[f'{var_seleccionado} hVaR Rolling'], label=f'{var_seleccionado} Rolling hVaR', color='black')
     ax_4.plot(CVaR_rolling_df.index, CVaR_rolling_df[f'{var_seleccionado} CVaR Rolling'], label=f'{var_seleccionado} Rolling CVaR', color='purple')
     ax_4.plot(hCVaR_rolling_df.index, hCVaR_rolling_df[f'{var_seleccionado} hCVaR Rolling'], label=f'{var_seleccionado} Rolling hCVaR', color='blue')

     ax_4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
     ax_4.legend()
     ax_4.set_title(f"Rendimientos de {activo_escogido[0]}")
     ax_4.set_xlabel("Fecha")
     ax_4.set_ylabel("Rendimiento Diario")
     st.pyplot(fig_4)



##Inciso D Vences
     rend = (df_rendimientos[activo_escogido[0]]*100).iloc[251:]
     B1 = VaR_rolling_df[f'{var_seleccionado} VaR Rolling'].iloc[251:]

     C1 = CVaR_rolling_df[f'{var_seleccionado} CVaR Rolling'].iloc[251:]

     v_var_1 = (rend < B1).sum()
     v_cvar_1 =(rend < C1).sum()

     p11=v_var_1/len(B1)
     p12=v_cvar_1/len(B1)


#print(f'Violaciones para Var 95% = {v_var_1} y un porcentaje de {(p11*100).round(4)}%')
#print(f'Violaciones para CVar 95% = {v_cvar_1} y un porcentaje de {(p12*100).round(4)}%')
#print(f'Violaciones para Var 97.5%= {v_var_2} y un porcentaje de {(p21*100).round(4)}%')
#print(f'Violaciones para CVar 97.5%= {v_cvar_2} y un porcentaje de {(p22*100).round(4)}%')
#print(f'Violaciones para Var 99% = {v_var_3} y un porcentaje de {(p31*100).round(4)}%')
#print(f'Violaciones para CVar 99% = {v_cvar_3} y un porcentaje de {(p32*100).round(4)}%')

# Diccionario con los datos

     
     data = {
               'Nivel de Confianza': [var_seleccionado,var_seleccionado],
          'Métrica': ['VaR','CVaR'],
          'Violaciones': [v_var_1, v_cvar_1, ],
          'Porcentaje (%)': [
               (p11 * 100).round(4),
               (p12 * 100).round(4)
          ]
          }

          # Crear el DataFrame
     df_violaciones = pd.DataFrame(data)

          # Mostrar el DataFrame
     st.dataframe(df_violaciones)
          
     
          
     
     
     st.header("VaR estimado con volatilidad móvil y asumiendo distribución normal")

     porcentaje_confianza = [0.95, 0.99]
     violaciones = []
     fig_5, ax_5 = plt.subplots(figsize=(14, 7))
     ax_5.plot(df_rendimientos[activo_escogido[0]] .index, df_rendimientos[activo_escogido[0]] * 100, label='Rendimientos Diarios (%)', color = 'blue', alpha = 0.5)

     for confianza in porcentaje_confianza:
          q_alpha = norm.ppf(1-confianza)
          rolling_std = df_rendimientos[activo_escogido[0]].rolling(window= 252).std()

          VaR_vm = rolling_std * q_alpha
          VaR_vm_percent = (VaR_vm * 100).round(4)
          VaR_vm_df = pd.DataFrame({'Date': df_rendimientos[activo_escogido[0]].index, f'{confianza} VaR Rolling': VaR_vm_percent.squeeze()})
          VaR_vm_df.set_index('Date', inplace=True)

          # Gráfico de rendimientos diarios y var volatil
          col = 'green'	
          if confianza == 0.95:
               col = 'blue'
          ax_5.plot(VaR_vm_df.index, VaR_vm_df[f'{confianza} VaR Rolling'], label=f'{confianza} VaR', color=col)

          # Unimos ambos DataFrames en base a la fecha
          df_merged = pd.merge(df_rendimientos[activo_escogido[0]] * 100 , VaR_vm_df[f'{confianza} VaR Rolling'], on='Date')
          # Idenfificamos las violaciones
          df_merged["Violacion_VaR"] = df_merged[activo_escogido[0]] < df_merged[f'{confianza} VaR Rolling']
          # Contamos ocurrencias
          conteo_violaciones = int(df_merged["Violacion_VaR"].sum())

          violaciones.append(conteo_violaciones)
          tam = (int(VaR_vm_df.count().sum()))





     ax_5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
     ax_5.legend()
     ax_5.set_title("Retornos Diarios y VaR")
     ax_5.set_xlabel("Fecha")
     ax_5.set_ylabel("Valores %")
     st.pyplot(fig_5)
     dt = pd.DataFrame({'Confianza': ['95 %', '99 %'], 'Violaciones de VaR': violaciones, 'Porcentaje de Violaciones de VaR': [100*x / tam for x in violaciones]})

     st.dataframe(dt, hide_index=True)        

