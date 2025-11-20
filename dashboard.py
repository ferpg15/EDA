import pandas as pd
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.cm as cm


st.set_page_config(
    page_title="Olist Dashboard",  
    page_icon="📦"                 
)
hide_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)


st.markdown("""
<style>

:root {
  --c1: #7AE582;
  --c2: #25A18E;
  --c3: #9FFFCB;
  --c4: #00A5CF;
  --c5: #004E64;
}

# .stApp {
#     background-color: #ffffff !important;
# }

section[data-testid="stSidebar"] {
    background-color: #f5f9f7 !important;
    border-right: 1px solid #e2e2e2;
}

h1, h2, h3 {
    color: var(--c5) !important;
    font-weight: 600;
}

section[data-testid="stSidebar"] * {
    color: var(--c5) !important;
}

div[role="radiogroup"] label {
    color: var(--c5) !important;
}

.stSlider > div > div > div {
    background: var(--c4) !important;
}
.stSlider > div > div > div > div {
    background: var(--c4) !important;
}


div[data-testid="stMetricValue"] {
    color: var(--c4) !important;
    font-weight: 700;
}


div[data-testid="stMetricLabel"] {
    color: var(--c5) !important;
}


.stButton>button {
    background-color: var(--c4);
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    border: none;
    transition: 0.2s;
}
.stButton>button:hover {
    background-color: var(--c5);
}

</style>
""", unsafe_allow_html=True)


# CARGA Y PREPARACIÓN DE DATOS

customers = pd.read_csv(
    "streamlit/customers.csv",
    parse_dates=['order_purchase_timestamp']
)

df_reviews = pd.read_csv(
    "resources/olist_order_reviews_dataset.csv"
)

customers_delivered = customers[customers['order_status'] == 'delivered']

customers_delivered['order_delivered_customer_date'] = pd.to_datetime(
    customers_delivered['order_delivered_customer_date']
)
customers_delivered['order_estimated_delivery_date'] = pd.to_datetime(
    customers_delivered['order_estimated_delivery_date']
)

customers_delivered['delay_days'] = (
    customers_delivered['order_delivered_customer_date'] -
    customers_delivered['order_estimated_delivery_date']
).dt.days
customers_delivered['late'] = customers_delivered['delay_days'] > 0

# SIDEBAR: FILTROS + NAVEGACIÓN

st.sidebar.title("Análisis Olist")

fecha_inicio = customers['order_purchase_timestamp'].min().date()
fecha_fin = customers['order_purchase_timestamp'].max().date()

st.sidebar.subheader("Filtros")

filtro_fecha = st.sidebar.slider(
    "Rango de fechas de compra:",
    value=(fecha_inicio, fecha_fin),
    min_value=fecha_inicio,
    max_value=fecha_fin
)

# Navegación entre páginas
pagina = st.sidebar.radio(
    "Ir a:",
    ["Inicio", "Clientes por estado", "Clientes por ciudad", "Análisis de retrasos", "Análisis de reviews"]
)


# APLICAR FILTROS

df_filtrado = customers[
    (customers['order_purchase_timestamp'].dt.date >= filtro_fecha[0]) &
    (customers['order_purchase_timestamp'].dt.date <= filtro_fecha[1])
].copy()

df_filtrado_delivered = customers_delivered[
    (customers_delivered['order_purchase_timestamp'].dt.date >= filtro_fecha[0]) &
    (customers_delivered['order_purchase_timestamp'].dt.date <= filtro_fecha[1])
].copy()


# FUNCIONES DE MÉTRICAS

def calcular_top_estados(df):
    
    df_estados = df.groupby('state')['id_user'].nunique().sort_values(ascending=False).head(5)
    
    df_estados = df_estados.reset_index()
    
    df_estados.rename(columns={'state': 'Estado', 'id_user': 'Total clientes'}, inplace=True)
    
    return df_estados

def calcular_ciudades(df):
    df_ciudades = (df.groupby(['city', 'state'])['id_user'].nunique().reset_index(name='Total clientes').sort_values('Total clientes', ascending=False))
    
    totalPedidos = df.groupby('city')['id_customer_order'].count().reset_index()
    totalPedidos.rename(columns={'id_customer_order': 'Pedidos totales'}, inplace=True)

    df_ciudades = pd.merge(df_ciudades, totalPedidos, on='city', how='left')

    df_ciudades['Porcentaje %'] = (
    df_ciudades['Pedidos totales'] / df_ciudades['Pedidos totales'].sum() * 100
    ).round(2)
    
    df_ciudades['Pedidos x cliente'] =(
        df_ciudades['Pedidos totales'] / df_ciudades['Total clientes']
    ).round(2)

    df_ciudades.rename(columns={'city': 'Ciudad', 'state':'Estado'}, inplace=True)

 
    return df_ciudades



def calcular_retrasos(df_delivered):
    customers_delivered = customers.query("order_status == 'delivered'").copy()
 
   
    customers_delivered['order_delivered_customer_date'] = pd.to_datetime(customers_delivered['order_delivered_customer_date'])
    customers_delivered['order_estimated_delivery_date'] = pd.to_datetime(customers_delivered['order_estimated_delivery_date'])
   
    customers_delivered['delay_days'] = (customers_delivered['order_delivered_customer_date'] - customers_delivered['order_estimated_delivery_date']).dt.days
    customers_delivered['late'] = customers_delivered['delay_days'] > 0
   
 
    pedidos_tarde = customers_delivered.groupby(["city", "state"]).agg(
        late=("late", "sum"),
        total_pedidos=("id_customer_order", "count"),
        days_late=("delay_days", lambda x: x[x > 0].mean() if any(x > 0) else 0)
    ).reset_index()
   
   
    pedidos_tarde["late_orders_%"] = (pedidos_tarde["late"] / pedidos_tarde["total_pedidos"] * 100).round(2)
 
 
   
    def diagnostico(row):
        if row['late_orders_%'] > 40 and row['days_late'] > 10:
            return "Problemas graves"
        elif row['late_orders_%'] > 25:
            return "Probable fallo del proveedor o mala preparación del pedido"
        elif row['late_orders_%'] > 15:
            return "Retrasos moderados (Posibles problemas con el repartidor)"
        else:
            return "Funcionamiento aceptable"
   
    pedidos_tarde['Diagnóstico'] = pedidos_tarde.apply(diagnostico, axis=1)
 
    pedidos_tarde = pedidos_tarde[["city", "state", "late_orders_%", "days_late", "Diagnóstico"]]
 
    pedidos_tarde.rename(columns={"city": "Ciudad", "state": "Estado", "late_orders_%": "Pedidos tarde %","days_late": "Dias tarde"   }, inplace=True)
    return pedidos_tarde



def calcular_reviews(customers_delivered, df_reviews):
    customers_review = customers_delivered[customers_delivered['late'] == False]

    customers_review = pd.merge(
        df_reviews,
        customers_review[['order_id','id_customer_order', 'id_user', 'state', 'order_purchase_timestamp']],
        on='order_id',
        how='left'
    )

    customers_review = customers_review.groupby(['state']).agg(
        Reviews=('order_id', 'count'),
        Puntuacion=('review_score', 'mean')
    ).reset_index()

    customers_review.rename(columns={'state': 'Estado'}, inplace=True)


    return customers_review




# Precalcular tablas
topEstados = calcular_top_estados(df_filtrado)
df_ciudades = calcular_ciudades(df_filtrado)
pedidos_tarde = calcular_retrasos(df_filtrado_delivered)
customers_review = calcular_reviews(customers_delivered, df_reviews)


# FUNCIONES GRÁFICOS


def grafico1(df_ciudades):
    
    top_users = df_ciudades.sort_values('Total clientes', ascending=False).head(10)

    st.title("Top 10 Ciudades por Número de Clientes")
        

    fig, ax = plt.subplots(figsize=(12, 6))
        
    bars = ax.bar(top_users['Ciudad'], top_users['Total clientes'], color='#004E64')
        
    ax.set_xlabel('Ciudad')
    ax.set_ylabel('Cantidad de Usuarios Únicos')
        

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        

    for i, v in enumerate(top_users['Total clientes']):
        ax.text(i, v + (v * 0.02), f"{v:,}", ha='center', fontsize=10)
        
    plt.tight_layout()
        

    st.pyplot(fig)



def grafico2(df_ciudades):
       
    top_orders = df_ciudades.sort_values('Pedidos totales', ascending=False).head(10)
    
    st.title("Top 10 Ciudades por Total de Pedidos")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(top_orders['Ciudad'], top_orders['Pedidos totales'], color="#4ca5b6")
    
    ax.set_xlabel('Ciudad')
    ax.set_ylabel('Total de Pedidos')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    for i, v in enumerate(top_orders['Pedidos totales']):
        ax.text(i, v + (v * 0.02), f"{v:,}", ha='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)




def grafico3(df_ciudades):
    color_users = "#4ca5b6"   
    color_orders = '#1d3557'  

    top_combined = df_ciudades.sort_values('Total clientes', ascending=False).head(10)
    

    st.title("Comparativa: Usuarios únicos vs Total de pedidos por ciudad (Top 10)")

    fig, ax = plt.subplots(figsize=(14, 6))
    

    ax.plot(
        top_combined['Ciudad'],
        top_combined['Total clientes'],
        marker='o',
        color=color_users,
        linewidth=2,
        label='Usuarios únicos'
    )
    

    ax.plot(
        top_combined['Ciudad'],
        top_combined['Pedidos totales'],
        marker='s',
        color=color_orders,
        linewidth=2,
        label='Total pedidos'
    )
    

    ax.set_xlabel('Ciudad')
    ax.set_ylabel('Cantidad')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    

    st.pyplot(fig)


def grafico4(df_ciudades):
    top_pct = df_ciudades.sort_values('Porcentaje %', ascending=False).head(10)
    st.title("Top 10 Ciudades por Porcentaje de Pedidos (%)")

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.bar(
        top_pct['Ciudad'],
        top_pct['Porcentaje %'],
        color='#52b788',     
    )
    
    ax.set_xlabel('Ciudad', fontsize=12)
    ax.set_ylabel('Porcentaje (%)', fontsize=12)
    

    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    

    for i, v in enumerate(top_pct['Porcentaje %']):
        ax.text(i, v + 0.5, f"{v}%", ha='center', fontsize=10)
    
    plt.tight_layout()
    

    st.pyplot(fig)


def grafico5(df_ciudades):
    top_pct = df_ciudades.sort_values('Porcentaje %', ascending=False).head(10)
    st.title("Participación de Pedidos por Ciudad (%) – Top 10")
    

    colors = [
        '#1d3557',  
        '#457b9d',
        '#6096ba',
        '#a8dadc',
        '#00b4d8',
        '#48cae4',
        '#90e0ef',
        '#52b788',  
        '#2d6a4f',
        '#95d5b2'   
    ]
    

    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.pie(
        top_pct['Porcentaje %'],
        labels=top_pct['Ciudad'],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    
    plt.tight_layout()

    st.pyplot(fig)


#GRAFICOS PAGINA 3
def grafico6(pedidos_tarde):
    top10 = pedidos_tarde.sort_values('Pedidos tarde %', ascending=False).head(10)
    
    cities = top10['Ciudad']
    days_late = top10['Dias tarde']
    
    st.title("Días Promedio de Retraso por Ciudad (Top 10)")
    

    fig, ax = plt.subplots(figsize=(12,6))
    

    bars = ax.bar(cities, days_late, color='#1d3557')
    
    ax.set_xlabel('Ciudad')
    ax.set_ylabel('Días')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    

    for i, v in enumerate(days_late):
        ax.text(i, v + 0.1, f"{v:.1f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    

    st.pyplot(fig)

def grafico7(pedidos_tarde):
    top10 = pedidos_tarde.sort_values('Pedidos tarde %', ascending=False)
 
    x = top10['Pedidos tarde %']
    y = top10['Dias tarde']
    cities = top10['Ciudad']
    

    coef = np.polyfit(x, y, 1)
    y_fit = np.polyval(coef, x)
    
    st.title("Relación: % Pedidos Tarde vs Días Promedio de Retraso")
    

    fig, ax = plt.subplots(figsize=(10,6))
    

    ax.scatter(x, y, color='#52b788', s=100, alpha=0.8, label='Ciudades')
    

    ax.plot(x, y_fit, color='#1d3557', linestyle='--', linewidth=2, label='Tendencia')
    

    ax.set_xlabel('% Pedidos entregados tarde')
    ax.set_ylabel('Días promedio de retraso')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

    correlacion = x.corr(y)
    st.write(f"**Coeficiente de correlación:** {correlacion:.2f}")


def grafico8(pedidos_tarde):
    top10 = pedidos_tarde.sort_values('Pedidos tarde %', ascending=False).head(10)
    cities = top10['Ciudad']
    
    x = np.arange(len(cities))
    width = 0.4
    
    st.title("Comparativa: Días promedio de retraso vs % de pedidos entregados tarde (Top 10 ciudades)")
    

    fig, ax1 = plt.subplots(figsize=(14,6))
    

    color_days_late = '#457b9d' 
    color_late_pct = '#2a9d8f'   
    

    bars1 = ax1.bar(x - width/2, top10['Dias tarde'], width, label='Días promedio de retraso', color=color_days_late)
    ax1.set_ylabel('Días promedio de retraso', color=color_days_late)
    ax1.tick_params(axis='y', labelcolor=color_days_late)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, top10['Pedidos tarde %'], width, label='% Pedidos entregados tarde', color=color_late_pct)
    ax2.set_ylabel('% Pedidos entregados tarde', color=color_late_pct)
    ax2.tick_params(axis='y', labelcolor=color_late_pct)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    

    st.pyplot(fig)

#GRAFICOS PAGINA 4
def grafico9(customers_review):
    customers_review_sorted = customers_review.sort_values('Puntuacion', ascending=False)
 
    st.title("Puntaje promedio de reviews por estado (Pedidos entregados a tiempo)")
    

    fig, ax = plt.subplots(figsize=(12,6))
    

    line_color = '#1d3557'  
    fill_color = '#a8dadc' 
    

    ax.plot(
        customers_review_sorted['Estado'],
        customers_review_sorted['Puntuacion'],
        marker='o',
        linestyle='-',
        color=line_color,
        linewidth=2,
        markersize=6,
        label='Puntaje promedio'
    )


    ax.set_xlabel('Estado', fontsize=12)
    ax.set_ylabel('Puntaje promedio', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)



def grafico10(customers_review):
    x = customers_review['Reviews']
    y = customers_review['Puntuacion']
    states = customers_review['Estado']
    
    st.title("Reviews vs Puntaje promedio por estado (Pedidos entregados a tiempo)")

    norm = (y - y.min()) / (y.max() - y.min())
    colors = cm.winter(norm)
    

    fig, ax = plt.subplots(figsize=(10,6))
    

    ax.scatter(x, y, color=colors, s=100, alpha=0.8)
    

    ax.set_xlabel('Número de reviews', fontsize=12)
    ax.set_ylabel('Puntaje promedio', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)






def grafico11(customers_review):
    customers_review_sorted = customers_review.sort_values('Puntuacion', ascending=False)
    
    st.title("Reviews por estado")

    heatmap_data = customers_review_sorted.pivot(
        index='Estado',
        columns='Reviews',
        values='Puntuacion'
    )
    
    fig, ax = plt.subplots(figsize=(10,8))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap='YlGnBu',
        cbar_kws={'label': 'Puntaje promedio'},
        ax=ax
    )
    
    ax.set_xlabel('Número de reviews')
    ax.set_ylabel('Estado')
    
    plt.tight_layout()
    
    st.pyplot(fig)



def mapa(geojson_path, df_pedidos):

    
    estados = gpd.read_file(geojson_path)

    df_pedidos = df_pedidos.reset_index()

    estados = estados.merge(
        df_pedidos[['Estado', 'Total clientes']],
        left_on='abbrev_state',
        right_on='Estado',
        how='left'
    )

    estados.rename(columns={'Total clientes': 'pedidos'}, inplace=True)

    estados['pedidos'] = estados['pedidos'].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    estados.plot(
        column="pedidos",
        cmap="viridis",
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=ax
    )
    ax.set_title("Mapa coroplético - Pedidos")
    ax.axis('off')  

    st.pyplot(fig)



# PÁGINA 0: INICIO (KPIs)

if pagina == "Inicio":
    st.image("olist.jpg")
    st.header("Dashboard Olist")

    st.write(f"**Rango de fechas aplicado:** {filtro_fecha[0]} — {filtro_fecha[1]}")

    col1, col2, col3, col4 = st.columns(4)

    total_pedidos = len(df_filtrado)
    clientes_unicos = df_filtrado['id_user'].nunique()

    if len(df_filtrado_delivered) > 0:
        porcentaje_tarde = (df_filtrado_delivered['late'].mean() * 100).round(2)
        retraso_medio = df_filtrado_delivered.loc[
            df_filtrado_delivered['delay_days'] > 0, 'delay_days'
        ].mean()
        retraso_medio = 0 if pd.isna(retraso_medio) else round(retraso_medio, 2)
    else:
        porcentaje_tarde = 0
        retraso_medio = 0

    col1.metric("Total pedidos", total_pedidos)
    col2.metric("Clientes únicos", clientes_unicos)
    col3.metric("% pedidos tarde", f"{porcentaje_tarde}%")
    col4.metric("Retraso medio (días)", retraso_medio)

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Top 5 estados por clientes")
        st.bar_chart(topEstados.set_index('Estado')['Total clientes'])

    with col6:
        st.subheader("Top 10 ciudades por pedidos")
        st.bar_chart(df_ciudades.set_index('Ciudad')['total_orders'].head(10))

    st.markdown("---")




# PÁGINA 1: CLIENTES POR ESTADO

elif pagina == "Clientes por estado":
    st.header("Clientes por estado")
    st.write(
        f"Rango de fechas: {filtro_fecha[0]} — {filtro_fecha[1]}"
    )

    st.subheader("Top 5 Clientes por estado")
    st.dataframe(topEstados)

    st.subheader("Top 5 Clientes por estado")
    st.bar_chart(topEstados.set_index('Estado')['Total clientes'])

    st.subheader("Mapa")
    mapa('br_states.geojson',topEstados)
    


# PÁGINA 2: CLIENTES POR CIUDAD

elif pagina == "Clientes por ciudad":
    st.header("Clientes por ciudad")
    st.write(
        f"Rango de fechas: {filtro_fecha[0]} — {filtro_fecha[1]}"
    )
 
    #FILTRO
 
    estados_disponibles = df_ciudades['Estado'].sort_values().unique()
    estado_seleccionado = st.selectbox("Filtrar por estado:", ["ALL STATES"] + list(estados_disponibles))
 
    if estado_seleccionado == 'ALL STATES':
        df_filtrado = df_ciudades
    else:
        df_filtrado = df_ciudades[df_ciudades['Estado'] == estado_seleccionado]
 
 
 
    st.subheader("Ranking de clientes por ciudades")
    st.dataframe(df_filtrado)
 
    grafico1(df_filtrado)
    grafico2(df_filtrado)
    grafico3(df_filtrado)
    grafico4(df_filtrado)
    grafico5(df_filtrado)
 




# PÁGINA 3: ANÁLISIS DE RETRASOS

elif pagina == "Análisis de retrasos":
    st.header("Análisis de retrasos")
    st.write(
          f"Rango de fechas: {filtro_fecha[0]} — {filtro_fecha[1]}"
    )
 
   
    #FILTRO
 
    estados_disponibles = pedidos_tarde['Estado'].sort_values().unique()
    estado_seleccionado = st.selectbox("Filtrar por estado:", ["ALL STATES"] + list(estados_disponibles))
 
    if estado_seleccionado == 'ALL STATES':
        pedidos_tarde_filtrado = pedidos_tarde
    else:
        pedidos_tarde_filtrado = pedidos_tarde[pedidos_tarde['Estado'] == estado_seleccionado]
 
 
 
    st.subheader("Revisión de demoras")
    st.dataframe(pedidos_tarde_filtrado)
 
    grafico6(pedidos_tarde_filtrado)
    grafico7(pedidos_tarde_filtrado)
    grafico8(pedidos_tarde_filtrado)


# PÁGINA 4: ANÁLISIS DE REVIEWS Y SCORE MEDIO
elif pagina == "Análisis de reviews":
    st.header("Reviews y score medio")
    st.write(
        f"Rango de fechas: {filtro_fecha[0]} — {filtro_fecha[1]}"
    )
    st.subheader("Análisis de reviews")
    st.dataframe(customers_review)
    grafico9(customers_review)
    grafico10(customers_review)
    grafico11(customers_review)
  