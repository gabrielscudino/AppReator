import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONFIGURACAO DA PAGINA ---
st.set_page_config(page_title="Simulador PFR", page_icon="🧪", layout="wide")

# Constante universal
Rjoule = 8.314

# ==========================================
# BARRA LATERAL (ENTRADAS DE DADOS)
# ==========================================
st.sidebar.title("🧪 Parametros do Reator")
st.sidebar.write("Simulacao em Fase Liquida")
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Condicoes de Operacao")
tempC = st.sidebar.number_input("Temperatura (°C)", value=25.0)
v0 = st.sidebar.number_input("Vazao Inicial (L/min)", value=10.0)
vtotal = st.sidebar.number_input("Volume do Reator (L)", value=100.0)

# Opcao de dimensionamento reverso
definirx = st.sidebar.checkbox("Calcular volume pela conversao alvo")
xalvo = 0.0
if definirx:
    xalvo = st.sidebar.number_input("Conversao desejada (0.01 a 0.99)", value=0.80, step=0.05)
    st.sidebar.caption("O volume digitado acima sera ignorado.")

tempK = tempC + 273.15

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Cinetica da Reacao")
tipok = st.sidebar.radio("Como definir a constante k?", ["Digitar o valor", "Usar Arrhenius"])

if tipok == "Digitar o valor":
    k = st.sidebar.number_input("Constante de velocidade (k)", value=0.05, format="%.4f")
else:
    A = st.sidebar.number_input("Fator A", value=1e7, format="%.2e")
    Ea = st.sidebar.number_input("Energia de Ativacao (Ea)", value=45000.0)
    k = A * np.exp(-Ea / (Rjoule * tempK))

st.sidebar.markdown("---")
st.sidebar.subheader("💧 Reagentes na Alimentacao")
numreagentes = st.sidebar.slider("Numero de Reagentes", 1, 2, 1)

reagentes = []
for i in range(numreagentes):
    st.sidebar.markdown(f"**Reagente {chr(65+i)}** {'(Limitante A)' if i==0 else ''}")
    col1, col2 = st.sidebar.columns(2)
    coef = col1.number_input("Coeficiente", value=1.0, key=f"coef{i}")
    c0 = col2.number_input("C0 (mol/L)", value=2.0 if i==0 else 1.0, key=f"c0{i}")
    reagentes.append({"coef": coef, "c0": c0})

simular = st.sidebar.button("Rodar Simulacao", type="primary", use_container_width=True)

st.sidebar.write("---")
st.sidebar.write("Desenvolvido por Gabriel Scudino Freitas")
st.sidebar.write("UFES - Projeto Computacional")


# ==========================================
# VERIFICACAO FISICA ANTES DO CALCULO
# ==========================================
ca0 = reagentes[0]["c0"]
coefA = reagentes[0]["coef"]

# Calcular qual é a conversao MAXIMA possivel de A antes de B acabar
xmaxpossivel = 1.0
if numreagentes == 2:
    cb0 = reagentes[1]["c0"]
    coefB = reagentes[1]["coef"]
    # Pela estequiometria, X maximo ocorre quando Cb = 0
    xmaxB = (cb0 * coefA) / (ca0 * coefB)
    if xmaxB < 1.0:
        xmaxpossivel = xmaxB

# Se o utilizador pedir algo impossivel, bloqueamos o calculo!
erroconversao = False
if definirx and xalvo > xmaxpossivel:
    erroconversao = True
    st.error(f"❌ **Erro Físico:** É impossível atingir {xalvo*100}% de conversão! O Reagente B vai esgotar-se quando a conversão de A atingir {xmaxpossivel*100:.1f}%. Aumente a concentração inicial de B ou diminua a conversão alvo.")


# ==========================================
# MOTOR MATEMATICO (ESTILO ESTUDANTE)
# ==========================================

if not erroconversao:
    
    # A equacao diferencial que vai para o solve_ivp
    def edopfr(v, xconv):
        conversao = xconv[0]
        
        if conversao >= 1.0:
            return [0.0]
        
        ca = ca0 * (1 - conversao)
        if ca <= 0.0:
            ca = 0.0
            
        # CORREÇÃO FÍSICA: Cinética Elementar (elevado ao coeficiente)
        ra = k * (ca ** coefA)
        
        if numreagentes == 2:
            cb = cb0 - (coefB / coefA) * (ca0 * conversao)
            if cb <= 0.0:
                cb = 0.0
                ra = 0.0 
            else:
                # CORREÇÃO FÍSICA: Cinética Elementar (elevado ao coeficiente)
                ra = ra * (cb ** coefB)
    
        fa0 = ca0 * v0
        if fa0 == 0:
            return [0.0]
            
        dxdv = ra / fa0
        return [dxdv]
    
    
    # Logica para descobrir o volume se o utilizador pediu
    if definirx and xalvo > 0:
        def atingiuconversao(v, xconv):
            return xconv[0] - xalvo
        atingiuconversao.terminal = True
        
        # Como garantimos que o alvo é possivel, ele vai achar o volume certo rapidamente
        restemp = solve_ivp(edopfr, [0, 1000000], [0.0], events=atingiuconversao)
        volusado = restemp.t[-1] 
    else:
        volusado = vtotal
    
    # Criar os pontos do eixo X 
    vspan = np.linspace(0, volusado, 200)
    
    # Resolucao definitiva da equacao
    resultado = solve_ivp(edopfr, [0, volusado], [0.0], t_eval=vspan)
    
    # Cortar errinhos matematicos do computador
    xres = np.clip(resultado.y[0], 0.0, 1.0)
    
    # Recalcula as concentracoes para o grafico 
    caperfil = np.maximum(ca0 * (1 - xres), 0.0)
    if numreagentes == 2:
        cbperfil = cb0 - (coefB / coefA) * (ca0 * xres)
        cbperfil = np.maximum(cbperfil, 0.0)


    # ==========================================
    # INTERFACE E GRAFICOS BONITOS
    # ==========================================
    
    st.title("Simulador de Reator Fluxo Pistao (PFR)")
    st.info("A simulacao decorre em **fase liquida** (volume constante) utilizando cinetica elementar baseada na quantidade de reagentes informados.")
    
    # Mostrar metricas numericas em destaque
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversao Final (X)", f"{xres[-1] * 100:.2f} %")
    c2.metric("Concentracao de A", f"{caperfil[-1]:.3f} mol/L")
    c3.metric("Volume do Reator", f"{volusado:.2f} L")
    c4.metric("Velocidade (k) Usada", f"{k:.4f}")
    
    st.write("---")
    st.subheader("Analise Grafica")
    
    colg1, colg2 = st.columns(2)
    
    plt.style.use('bmh') 
    
    with colg1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(vspan, xres * 100, color='#E63946', linewidth=3)
        ax1.fill_between(vspan, xres * 100, color='#E63946', alpha=0.1)
        ax1.set_title("Evolucao da Conversao", fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel("Volume do Reator (L)", fontsize=10)
        ax1.set_ylabel("Conversao (%)", fontsize=10)
        ax1.set_ylim(-2, 105) 
        st.pyplot(fig1)
    
    with colg2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(vspan, caperfil, color='#1D3557', linewidth=2.5, label='Reagente A')
        
        if numreagentes == 2:
            ax2.plot(vspan, cbperfil, color='#457B9D', linewidth=2.5, label='Reagente B')
            
        ax2.set_title("Perfil de Concentracoes", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Volume do Reator (L)", fontsize=10)
        ax2.set_ylabel("Concentracao (mol/L)", fontsize=10)
        ax2.set_ylim(bottom=-0.1) 
        ax2.legend()
        st.pyplot(fig2)
    
    # Construcao da tabela de dados
    mostrartabela = st.checkbox("Mostrar tabela de valores amostrados", value=True)
    
    if mostrartabela:
        st.write("---")
        
        dados = {
            "Volume (L)": vspan,
            "Conversao (X)": xres,
            "Conc. A (mol/L)": caperfil
        }
        if numreagentes == 2:
            dados["Conc. B (mol/L)"] = cbperfil
            
        df = pd.DataFrame(dados)
        
        dfresumo = df.iloc[::10, :].copy()
        dfresumo = pd.concat([dfresumo, df.iloc[[-1]]]).drop_duplicates()
        
        formatacao = {"Volume (L)": "{:.2f}", "Conversao (X)": "{:.4f}", "Conc. A (mol/L)": "{:.4f}"}
        if numreagentes == 2: 
            formatacao["Conc. B (mol/L)"] = "{:.4f}"
            
        st.dataframe(dfresumo.style.format(formatacao), use_container_width=True)
    
    if simular:
        st.success("✅ Simulacao calculada com sucesso!")
