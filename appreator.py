import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Simulador PFR", page_icon="🧪", layout="wide")

# Constante universal
Rjoule = 8.314

# ==========================================
# BARRA LATERAL (ENTRADAS DE DADOS)
# ==========================================
st.sidebar.title("🧪 Parâmetros do Reator")
st.sidebar.write("Simulação em Fase Líquida")
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Condições de Operação")
tempC = st.sidebar.number_input("Temperatura (°C)", value=25.0)
v0 = st.sidebar.number_input("Vazão Inicial (L/min)", value=10.0)
vtotal = st.sidebar.number_input("Volume do Reator (L)", value=100.0)

# Opção de dimensionamento reverso
definirx = st.sidebar.checkbox("Calcular volume pela conversão alvo")
xalvo = 0.0
if definirx:
    xalvo = st.sidebar.number_input("Conversão desejada (0.01 a 0.99)", value=0.80, step=0.05)
    st.sidebar.caption("O volume digitado acima será ignorado.")

tempK = tempC + 273.15

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Cinética da Reação")
tipok = st.sidebar.radio("Como definir a constante k?", ["Digitar o valor", "Usar Arrhenius"])

if tipok == "Digitar o valor":
    k = st.sidebar.number_input("Constante de velocidade (k)", value=0.05, format="%.4f")
else:
    A = st.sidebar.number_input("Fator A", value=1e7, format="%.2e")
    Ea = st.sidebar.number_input("Energia de Ativação (Ea)", value=45000.0)
    k = A * np.exp(-Ea / (Rjoule * tempK))

st.sidebar.markdown("---")
st.sidebar.subheader("💧 Reagentes na Alimentação")
# Limitado agora a no máximo 2 reagentes
num_reagentes = st.sidebar.slider("Número de Reagentes", 1, 2, 1)

reagentes = []
for i in range(num_reagentes):
    st.sidebar.markdown(f"**Reagente {chr(65+i)}** {'(Limitante)' if i==0 else ''}")
    col1, col2 = st.sidebar.columns(2)
    coef = col1.number_input("Coeficiente", value=1.0, key=f"coef{i}")
    c0 = col2.number_input("C0 (mol/L)", value=2.0 if i==0 else 1.0, key=f"c0{i}")
    reagentes.append({"coef": coef, "c0": c0})

simular = st.sidebar.button("Rodar Simulação", type="primary", use_container_width=True)

st.sidebar.write("---")
st.sidebar.write("Desenvolvido por Gabriel Scudino Freitas")
st.sidebar.write("UFES - Projeto Computacional")


# ==========================================
# MOTOR MATEMÁTICO (ESTILO ESTUDANTE)
# ==========================================

# Pegar os dados do reagente limitante (sempre o primeiro, Reagente A)
ca0 = reagentes[0]["c0"]
coefA = reagentes[0]["coef"]

# A equação diferencial que vai para o solve_ivp
def edopfr(v, x_conv):
    conversao = x_conv[0]
    
    # 1. Calcular a concentração do Reagente A (Limitante)
    ca = ca0 * (1 - conversao)
    
    # 2. Base da taxa de reação
    ra = k * ca
    
    # 3. Incluir a concentração do segundo reagente na taxa (se existir)
    if num_reagentes == 2:
        cb0 = reagentes[1]["c0"]
        coefB = reagentes[1]["coef"]
        
        # Balanço estequiométrico simples para o reagente B
        cb = cb0 - (coefB / coefA) * (ca0 * conversao)
        cb = max(cb, 0.0) # Evitar que fique negativo matematicamente
        ra = ra * cb

    # 4. Finalizar a EDO dX/dV
    fa0 = ca0 * v0
    dxdv = ra / fa0
    
    return [dxdv]


# Lógica para descobrir o volume se o usuário pediu
if definirx and xalvo > 0:
    # Função evento: o solve_ivp vai parar exatamente quando a conversão atingir o alvo
    def atingiu_conversao(v, x_conv):
        return x_conv[0] - xalvo
    atingiu_conversao.terminal = True
    
    # Roda a simulação até um volume gigante (100.000), mas ele para sozinho no momento certo
    res_temp = solve_ivp(edopfr, [0, 100000], [0.0], events=atingiu_conversao)
    volusado = res_temp.t[-1] # Pega o exato volume onde a simulação parou
else:
    volusado = vtotal

# Criar os pontos do eixo X para desenhar os gráficos de forma suave
vspan = np.linspace(0, volusado, 200)

# Resolução definitiva da equação
resultado = solve_ivp(edopfr, [0, volusado], [0.0], t_eval=vspan)
xres = resultado.y[0]

# Calcular os vetores de concentração para gerar as curvas do gráfico
ca_perfil = ca0 * (1 - xres)
if num_reagentes == 2:
    cb_perfil = reagentes[1]["c0"] - (reagentes[1]["coef"] / coefA) * (ca0 * xres)
    cb_perfil = np.maximum(cb_perfil, 0.0)


# ==========================================
# INTERFACE E GRÁFICOS BONITOS
# ==========================================

st.title("Simulador de Reator Fluxo Pistão (PFR)")
st.info("A simulação decorre em **fase líquida** (volume constante) utilizando cinética elementar baseada na quantidade de reagentes informados.")

# Mostrar métricas numéricas em destaque
c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversão Final (X)", f"{xres[-1] * 100:.2f} %")
c2.metric("Concentração de A", f"{ca_perfil[-1]:.3f} mol/L")
c3.metric("Volume do Reator", f"{volusado:.2f} L")
c4.metric("Velocidade (k) Usada", f"{k:.4f}")

st.write("---")
st.subheader("Análise Gráfica")

colg1, colg2 = st.columns(2)

# Configurações estéticas (Deixando o gráfico com aparência profissional)
plt.style.use('bmh') 

with colg1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(vspan, xres * 100, color='#E63946', linewidth=3) # Vermelho elegante
    ax1.fill_between(vspan, xres * 100, color='#E63946', alpha=0.1) # Sombra em baixo da curva
    ax1.set_title("Evolução da Conversão", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Volume do Reator (L)", fontsize=10)
    ax1.set_ylabel("Conversão (%)", fontsize=10)
    st.pyplot(fig1)

with colg2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(vspan, ca_perfil, color='#1D3557', linewidth=2.5, label='Reagente A') # Azul escuro
    
    # Se houver 2 reagentes, plota a linha do B no mesmo gráfico
    if num_reagentes == 2:
        ax2.plot(vspan, cb_perfil, color='#457B9D', linewidth=2.5, label='Reagente B') # Azul claro
        
    ax2.set_title("Perfil de Concentrações", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Volume do Reator (L)", fontsize=10)
    ax2.set_ylabel("Concentração (mol/L)", fontsize=10)
    ax2.legend()
    st.pyplot(fig2)


# Construção da tabela de dados
mostrartabela = st.checkbox("Mostrar tabela de valores amostrados", value=True)

if mostrartabela:
    st.write("---")
    
    dados = {
        "Volume (L)": vspan,
        "Conversão (X)": xres,
        "Conc. A (mol/L)": ca_perfil
    }
    if num_reagentes == 2:
        dados["Conc. B (mol/L)"] = cb_perfil
        
    df = pd.DataFrame(dados)
    
    # Resume a tabela para não poluir o ecrã
    df_resumo = df.iloc[::10, :].copy()
    df_resumo = pd.concat([df_resumo, df.iloc[[-1]]]).drop_duplicates()
    
    # Formatação das casas decimais da tabela
    formatacao = {"Volume (L)": "{:.2f}", "Conversão (X)": "{:.4f}", "Conc. A (mol/L)": "{:.4f}"}
    if num_reagentes == 2: 
        formatacao["Conc. B (mol/L)"] = "{:.4f}"
        
    st.dataframe(df_resumo.style.format(formatacao), use_container_width=True)

if simular:
    st.success("✅ Simulação calculada com sucesso!")
