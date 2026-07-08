import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Configuracao simples da pagina
st.set_page_config(page_title="Simulador PFR Liquido", layout="wide")

# Constante dos gases
Rjoule = 8.314

# --- BARRA LATERAL (ENTRADAS) ---
st.sidebar.title("Dados do Problema")

st.sidebar.markdown("### Condicoes de Operacao")
tempC = st.sidebar.number_input("Temperatura (C)", value=25.0)
v0 = st.sidebar.number_input("Vazao (L/min)", value=10.0)
vtotal = st.sidebar.number_input("Volume do Reator (L)", value=100.0)

# Opcao para calcular o volume pela conversao
definirx = st.sidebar.checkbox("Quero calcular o volume pela conversao")
if definirx:
    xalvo = st.sidebar.number_input("Conversao desejada (0 a 0.99)", value=0.80)
    st.sidebar.write("O volume digitado em cima vai ser ignorado.")
else:
    xalvo = 0

tempK = tempC + 273.15

st.sidebar.markdown("### Cinetica")
tipok = st.sidebar.radio("Como queres o k?", ["Digitar k", "Usar Arrhenius"])

if tipok == "Digitar k":
    k = st.sidebar.number_input("Valor de k", value=0.05, format="%.4f")
else:
    A = st.sidebar.number_input("Fator A", value=1e7, format="%.2e")
    Ea = st.sidebar.number_input("Energia Ea (J/mol)", value=45000.0)
    k = A * np.exp(-Ea / (Rjoule * tempK))

st.sidebar.markdown("### Reagente")
ca0 = st.sidebar.number_input("Concentracao de A (mol/L)", value=2.00)

simular = st.sidebar.button("Simular")


# --- RESOLUCAO MATEMATICA ---

# Equacao diferencial para o solve_ivp
# Nota: o solve_ivp pede a ordem (variavel independente, variavel dependente)
def edopfr(v, x, k_reacao, ca_zero, vazao):
    # O solve_ivp trata o x como um array, por isso usamos x[0]
    conversao = x[0] 
    
    # Concentracao em fase liquida (volume constante)
    ca = ca_zero * (1 - conversao)
    
    # Taxa de reacao
    ra = k_reacao * ca
    fa0 = ca_zero * vazao
    
    # dxdv
    dxdv = ra / fa0
    return [dxdv]

# Verifica se é para calcular o volume analiticamente
if definirx and xalvo > 0:
    # Formula integrada para 1a ordem em fase liquida
    volcalc = (v0 / k) * np.log(1 / (1 - xalvo))
    volusado = volcalc
else:
    volusado = vtotal

# Criar os pontos de volume para o grafico
vspan = np.linspace(0, volusado, 200)

# Resolver a EDO com solve_ivp
resultado = solve_ivp(
    fun=edopfr, 
    t_span=[0, volusado], 
    y0=[0.0], 
    t_eval=vspan, 
    args=(k, ca0, v0)
)

# O resultado da conversao fica guardado em resultado.y
xres = resultado.y[0]

# Calcular a concentracao final em cada ponto
caperfil = ca0 * (1 - xres)


# --- MOSTRAR RESULTADOS NO ECRÃ ---

st.title("Simulador de Reator PFR (Fase Líquida)")
st.write("Simulação de um reator fluxo pistão para líquidos, assumindo volume constante.")

st.write(f"**Constante cinética (k):** {k:.4f}")

# Mostrar resultados finais
col1, col2, col3 = st.columns(3)
col1.metric("Conversao de Saida (X)", f"{xres[-1] * 100:.2f} %")
col2.metric("Concentracao Final (Ca)", f"{caperfil[-1]:.3f} mol/L")
col3.metric("Volume do Reator", f"{volusado:.2f} L")

st.write("---")

# Fazer os graficos
colg1, colg2 = st.columns(2)

with colg1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(vspan, xres * 100, color='red', linewidth=2)
    ax1.set_title("Conversao")
    ax1.set_xlabel("Volume (L)")
    ax1.set_ylabel("X (%)")
    ax1.grid(True)
    st.pyplot(fig1)

with colg2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(vspan, caperfil, color='blue', linewidth=2)
    ax2.set_title("Concentracao de A")
    ax2.set_xlabel("Volume (L)")
    ax2.set_ylabel("Ca (mol/L)")
    ax2.grid(True)
    st.pyplot(fig2)

# Fazer a tabela simples
mostrartabela = st.checkbox("Mostrar tabela de valores")

if mostrartabela:
    st.write("Tabela de dados:")
    
    tabela = pd.DataFrame({
        "Volume (L)": vspan,
        "Conversao": xres,
        "Concentracao (mol/L)": caperfil
    })
    
    # Pega apenas alguns valores para nao ficar gigante
    tabelaresumo = tabela.iloc[::15, :].copy()
    
    st.dataframe(tabelaresumo)

if simular:
    st.success("Conta feita com sucesso!")
