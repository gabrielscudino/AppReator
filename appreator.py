import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Simulador PFR", page_icon="🧪", layout="wide")

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
numreagentes = st.sidebar.slider("Número de Reagentes", 1, 2, 1)

reagentes = []
for i in range(numreagentes):
    st.sidebar.markdown(f"**Reagente {chr(65+i)}** {'(Limitante A)' if i==0 else ''}")
    col1, col2 = st.sidebar.columns(2)
    coef = col1.number_input("Coeficiente", value=1.0, key=f"coef{i}")
    c0 = col2.number_input("C0 (mol/L)", value=2.0 if i==0 else 1.0, key=f"c0{i}")
    reagentes.append({"coef": coef, "c0": c0})

simular = st.sidebar.button("Rodar Simulação", type="primary", use_container_width=True)

st.sidebar.write("---")
st.sidebar.write("Desenvolvido por Gabriel Scudino Freitas")
st.sidebar.write("UFES - Projeto Computacional")


# ==========================================
# VERIFICAÇÃO FÍSICA ANTES DO CÁLCULO
# ==========================================
ca0 = reagentes[0]["c0"]
coefA = reagentes[0]["coef"]

# Calcular qual é a conversão MÁXIMA possível de A antes de B acabar
xmaxpossivel = 1.0
if numreagentes == 2:
    cb0 = reagentes[1]["c0"]
    coefB = reagentes[1]["coef"]
    # Pela estequiometria, X máximo ocorre quando Cb = 0
    xmaxB = (cb0 * coefA) / (ca0 * coefB)
    if xmaxB < 1.0:
        xmaxpossivel = xmaxB

# Lógica de verificação contra valores impossíveis
erroconversao = False
if definirx and xalvo > xmaxpossivel:
    erroconversao = True
    st.error(f"❌ **Erro Físico:** É impossível atingir {xalvo*100}% de conversão! O Reagente B vai esgotar-se quando a conversão de A atingir {xmaxpossivel*100:.1f}%. Aumente a concentração inicial de B ou diminua a conversão alvo.")


# ==========================================
# MOTOR MATEMÁTICO (ESTILO ESTUDANTE)
# ==========================================

if not erroconversao:
    
    # A equação diferencial que vai para o solve_ivp
    def edopfr(v, xconv):
        conversao = xconv[0]
        
        if conversao >= 1.0:
            return [0.0]
        
        ca = ca0 * (1 - conversao)
        if ca <= 0.0:
            ca = 0.0
            
        ra = k * (ca ** coefA)
        
        if numreagentes == 2:
            cb = cb0 - (coefB / coefA) * (ca0 * conversao)
            if cb <= 0.0:
                cb = 0.0
                ra = 0.0 
            else:
                ra = ra * (cb ** coefB)
    
        fa0 = ca0 * v0
        if fa0 == 0:
            return [0.0]
            
        dxdv = ra / fa0
        return [dxdv]
    
    
    # Lógica para descobrir o volume se o utilizador pediu
    if definirx and xalvo > 0:
        def atingiuconversao(v, xconv):
            return xconv[0] - xalvo
        atingiuconversao.terminal = True
        
        # O programa procura o volume exato onde o alvo é atingido
        restemp = solve_ivp(edopfr, [0, 1000000], [0.0], events=atingiuconversao)
        volusado = restemp.t[-1] 
    else:
        volusado = vtotal
    
    # Criar os pontos do eixo X 
    vspan = np.linspace(0, volusado, 200)
    
    # Resolução definitiva da equação
    resultado = solve_ivp(edopfr, [0, volusado], [0.0], t_eval=vspan)
    
    # Cortar errinhos matemáticos do computador
    xres = np.clip(resultado.y[0], 0.0, 1.0)
    
    # Recalcula as concentrações para o gráfico 
    caperfil = np.maximum(ca0 * (1 - xres), 0.0)
    if numreagentes == 2:
        cbperfil = cb0 - (coefB / coefA) * (ca0 * xres)
        cbperfil = np.maximum(cbperfil, 0.0)
        
    
    # ==========================================
    # INTERFACE PRINCIPAL (ABAS)
    # ==========================================
    
    st.title("Simulador de Reator Fluxo Pistão (PFR)")
    
    # Criação das duas abas principais
    aba1, aba2 = st.tabs(["📊 Simulador", "📖 Entendendo o Reator"])
    
    with aba1:
        st.info("A simulação decorre em **fase líquida** (volume constante) utilizando cinética elementar baseada na quantidade de reagentes informados.")
        
        # Mostrar métricas numéricas em destaque
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Conversão Final (X)", f"{xres[-1] * 100:.2f} %")
        c2.metric("Concentração de A", f"{caperfil[-1]:.3f} mol/L")
        c3.metric("Volume do Reator", f"{volusado:.2f} L")
        c4.metric("Velocidade (k) Usada", f"{k:.4f}")
        
        st.write("---")
        st.subheader("Análise Gráfica")
        
        colg1, colg2 = st.columns(2)
        plt.style.use('bmh') 
        
        with colg1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(vspan, xres * 100, color='#E63946', linewidth=3)
            ax1.fill_between(vspan, xres * 100, color='#E63946', alpha=0.1)
            ax1.set_title("Evolução da Conversão", fontsize=12, fontweight='bold', pad=10)
            ax1.set_xlabel("Volume do Reator (L)", fontsize=10)
            ax1.set_ylabel("Conversão (%)", fontsize=10)
            ax1.set_ylim(-2, 105) 
            st.pyplot(fig1)
        
        with colg2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(vspan, caperfil, color='#1D3557', linewidth=2.5, label='Reagente A')
            
            if numreagentes == 2:
                ax2.plot(vspan, cbperfil, color='#457B9D', linewidth=2.5, label='Reagente B')
                
            ax2.set_title("Perfil de Concentrações", fontsize=12, fontweight='bold', pad=10)
            ax2.set_xlabel("Volume do Reator (L)", fontsize=10)
            ax2.set_ylabel("Concentração (mol/L)", fontsize=10)
            ax2.set_ylim(bottom=-0.1) 
            ax2.legend()
            st.pyplot(fig2)
        
        # Construção da tabela de dados
        mostrartabela = st.checkbox("Mostrar tabela de valores amostrados", value=True)
        
        if mostrartabela:
            st.write("---")
            
            dados = {
                "Volume (L)": vspan,
                "Conversão (X)": xres,
                "Conc. A (mol/L)": caperfil
            }
            if numreagentes == 2:
                dados["Conc. B (mol/L)"] = cbperfil
                
            df = pd.DataFrame(dados)
            
            dfresumo = df.iloc[::10, :].copy()
            dfresumo = pd.concat([dfresumo, df.iloc[[-1]]]).drop_duplicates()
            
            formatacao = {"Volume (L)": "{:.2f}", "Conversão (X)": "{:.4f}", "Conc. A (mol/L)": "{:.4f}"}
            if numreagentes == 2: 
                formatacao["Conc. B (mol/L)"] = "{:.4f}"
                
            st.dataframe(dfresumo.style.format(formatacao), use_container_width=True)

    with aba2:
        st.header("Análise Teórica do Reator PFR")
        st.write("""
        O Reator de Fluxo Pistão (PFR - *Plug Flow Reactor*) é um modelo ideal de reator tubular. 
        Neste modelo, assume-se que o fluido se move em forma de "pistão", ou seja, a mistura é perfeita na direção radial (perpendicular ao fluxo), mas não há nenhuma mistura na direção axial (ao longo do tubo).
        """)
        
        st.subheader("1. Balanço de Massa")
        st.write("A equação de projeto fundamental para um PFR, que descreve a variação da conversão ($X$) ao longo do volume ($V$) do reator, é dada por:")
        st.latex(r" \frac{dX}{dV} = \frac{-r_A}{F_{A0}} ")
        st.write("""
        Onde:
        * $X$ é a conversão fracional do reagente limitante A.
        * $V$ é o volume do reator (L).
        * $F_{A0}$ é a vazão molar inicial de A, calculada pela multiplicação da concentração inicial pela vazão volumétrica ($F_{A0} = C_{A0} \cdot v_0$).
        * $-r_A$ é a taxa de reação de consumo de A.
        """)
        
        st.subheader("2. Estequiometria e Concentração")
        st.write("Como este simulador opera em **fase líquida**, assumimos que a densidade do fluido é constante ao longo de todo o processo. Portanto, a vazão volumétrica inicial ($v_0$) não se altera. As concentrações dos reagentes diminuem estritamente devido ao consumo químico:")
        st.latex(r" C_A = C_{A0}(1 - X) ")
        
        if numreagentes == 2:
            st.write("Para o reagente secundário B, a concentração é calculada respeitando a proporção estequiométrica:")
            st.latex(r" C_B = C_{B0} - \frac{\nu_B}{\nu_A}(C_{A0} \cdot X) ")
            st.write("Onde $\\nu_A$ e $\\nu_B$ são os coeficientes estequiométricos.")
        
        st.subheader("3. Cinética da Reação")
        st.write("O simulador utiliza um modelo de **Cinética Elementar**. Isto significa que a ordem da reação em relação a cada reagente é idêntica ao seu coeficiente estequiométrico. A lei de velocidade é definida por:")
        if numreagentes == 1:
            st.latex(r" -r_A = k \cdot C_A^{\nu_A} ")
        else:
            st.latex(r" -r_A = k \cdot C_A^{\nu_A} \cdot C_B^{\nu_B} ")
            
        st.write("A constante de velocidade ($k$) pode ser fixa ou calculada dependendo da temperatura utilizando a **Equação de Arrhenius**:")
        st.latex(r" k = A \cdot e^{\frac{-E_a}{R \cdot T}} ")
        
        st.subheader("4. Resolução Numérica")
        st.write("""
        Matematicamente, não é possível encontrar uma solução algébrica simples (isolando o $X$) para reações com múltiplos reagentes e ordens diferentes. 
        Por isso, este simulador utiliza o método numérico computacional **Runge-Kutta** (através da função `solve_ivp` da biblioteca SciPy do Python) para fatiar o volume do reator em pequenos pedaços e integrar a equação diferencial passo a passo desde a entrada ($V=0$) até à saída ($V=V_{total}$).
        """)
