import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Configuracoes da pagina (comandos obrigatorios da biblioteca)
st.set_page_config(
    page_title="Simulador de Reator PFR",
    page_icon="🧪",
    layout="wide"
)

# Constantes Universais de Engenharia Quimica
Rjoule = 8.314     # Usada na Equacao de Arrhenius
Ratm = 0.08206     # Usada na Lei dos Gases Ideais

# Painel lateral de controle para entrada de parametros
st.sidebar.title("Parametros de Entrada")

# Selecao de fase do sistema
fasesistema = st.sidebar.selectbox("Fase do Sistema", ["Líquida", "Gasosa"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Condicoes de Operacao")

# Variaveis operacionais por digitacao direta
tempC = st.sidebar.number_input("Temperatura de Operacao (°C)", value=25.0, step=5.0, format="%.1f")
patm = st.sidebar.number_input("Pressao de Operacao (atm)", min_value=0.1, value=1.0, step=0.1, format="%.2f")
v0 = st.sidebar.number_input("Vazao Volumétrica Inicial (L/min)", min_value=0.1, value=10.0, step=1.0)
vtotal = st.sidebar.number_input("Volume total do Reator PFR (L)", min_value=1.0, value=100.0, step=10.0)

# CONVERSAO ALVO OPTATIVA: Se marcar, calcula o Volume. Se desmarcar, calcula a Conversao.
definirXalvo = st.sidebar.checkbox("Definir Conversao Alvo (Calcular Volume)", value=False)
if definirXalvo:
    Xalvo = st.sidebar.number_input("Conversao Alvo Desejada (X de 0 a 0.999)", min_value=0.0, max_value=0.999, value=0.80, step=0.05)
    st.sidebar.caption("💡 O volume total informado acima sera ignorado e recalculado pelo sistema.")
else:
    Xalvo = None

# Conversao da temperatura para Kelvin
tempK = tempC + 273.15

st.sidebar.markdown("---")
st.sidebar.markdown("### Cinetica da Reacao")

# Escolha de como inserir a velocidade da reacao
metodok = st.sidebar.radio(
    "Definicao da Constante de Velocidade (k)",
    ["Digitar valor direto de k", "Calcular via Arrhenius"]
)

if metodok == "Digitar valor direto de k":
    kfinal = st.sidebar.number_input("Constante de Velocidade (k)", min_value=0.0001, value=0.05, format="%.4f")
else:
    Apre = st.sidebar.number_input("Fator Pre-Exponencial (A)", min_value=1.0, value=1e7, format="%.2e")
    Ea = st.sidebar.number_input("Energia de Ativacao (Ea) [J/mol]", min_value=0.0, value=45000.0, step=1000.0)
    kfinal = Apre * np.exp(-Ea / (Rjoule * tempK))

st.sidebar.markdown("---")
st.sidebar.markdown("### Estequiometria e Alimentacao")

# Calculo de gases ideais
if fasesistema == "Gasosa":
    ctotal0 = patm / (Ratm * tempK)
else:
    ctotal0 = None

# Inserção dinamica de Reagentes
numreagentes = st.sidebar.number_input("Quantidade de Reagentes", min_value=1, max_value=5, value=1, step=1)
reagentes = []
somay0 = 0.0
dadosinvalidos = False

for i in range(int(numreagentes)):
    st.sidebar.markdown(f"**Reagente {i+1}** " + ("*(Reagente Limitante A)*" if i == 0 else ""))
    col1, col2 = st.sidebar.columns(2)
    coef = col1.number_input("Coef. Esteq.", min_value=0.1, value=1.0, step=0.5, key=f"coefr{i}")
    
    if fasesistema == "Gasosa":
        rotuloy0 = "Fracao Molar (yA0)" if i == 0 else f"Fracao Molar (y0)"
        valpadrao = 1.0 / numreagentes
        
        y0 = col2.number_input(rotuloy0, min_value=0.0, max_value=1.0, value=valpadrao, step=0.05, key=f"y0r{i}")
        somay0 += y0
        c0 = y0 * ctotal0
    else:
        c0 = col2.number_input("C0 (mol/L)", min_value=0.01, value=2.00, step=0.1, key=f"c0r{i}")
        y0 = 0.0
    
    reagentes.append({"coef": coef, "c0": c0, "y0": y0})

if fasesistema == "Gasosa":
    if somay0 > 1.0:
        st.sidebar.error(f"❌ Erro: A soma das fracoes molares superou 1.0!")
        dadosinvalidos = True
    else:
        yinertes = 1.0 - somay0
        if yinertes > 0.001:
            st.sidebar.info(f"💡 Fracao de gases inertes: {yinertes:.3f}")

# Inserção dinamica de Produtos
numprodutos = st.sidebar.number_input("Quantidade de Produtos", min_value=1, max_value=5, value=1, step=1)
produtos = []

for i in range(int(numprodutos)):
    coef = st.sidebar.number_input(f"Coef. Esteq. Produto {i+1}", min_value=0.1, value=1.0, step=0.5, key=f"coefp{i}")
    produtos.append({"coef": coef})

mostrartabela = st.sidebar.checkbox("Mostrar tabela de valores intermediarios", value=True)
simular = st.sidebar.button("Simular")

st.sidebar.write("---")
st.sidebar.write("Desenvolvido por Gabriel Scudino Freitas")
st.sidebar.write("email: gabriel.freitas.00@edu.ufes.br")


# ==========================================
# MOTOR MATEMATICO DE RESOLUÇÃO (ROBUSTO)
# ==========================================

if dadosinvalidos:
    st.error("Corrija os erros de fracao molar para rodar o simulador.")
else:
    ca0 = reagentes[0]["c0"]
    coefA = reagentes[0]["coef"]
    ya0 = reagentes[0]["y0"] if fasesistema == "Gasosa" else 0.0

    # Calculo estequiometrico global do delta e expansao epsilon
    somacoefreagentes = sum([r["coef"] for r in reagentes])
    somacoefprodutos = sum([p["coef"] for p in produtos])
    delta = (somacoefprodutos - somacoefreagentes) / coefA
    epsilon = ya0 * delta if fasesistema == "Gasosa" else 0.0

    # Funcao com a equacao diferencial (EDO) de projeto do PFR
    def pfrsystem(X, V, k, ca0val, v0val, eps):
        fa0 = ca0val * v0val
        vlocal = v0val * (1 + eps * X)
        vlocal = max(vlocal, 1e-6) 
        ca = (ca0val * (1 - X)) / (1 + eps * X)
        ca = max(ca, 0.0)
        ra = k * ca
        dxdv = ra / fa0
        return dxdv

    # CONDICIONAL DA INCÓGNITA: Verifica qual modo de calculo rodar
    if definirXalvo and Xalvo > 0:
        # Modo Reverso: O usuario deu a conversao, calculamos o volume por integracao analitica de 1a ordem
        termoLn = np.log(1 / (1 - Xalvo))
        Vcalculado = (v0 / kfinal) * ((1 + epsilon) * termoLn - epsilon * Xalvo)
        vtotalUsado = Vcalculado
    else:
        # Modo Direto Normal: Usa o volume fixo digitado pelo usuario
        vtotalUsado = vtotal

    # Define a malha de integracao ate o volume correto (recalculado ou fixo)
    vspan = np.linspace(0, vtotalUsado, 200)
    X0 = 0.0

    # Resolucao numerica final do perfil axial
    Xres = odeint(pfrsystem, X0, vspan, args=(kfinal, ca0, v0, epsilon)).flatten()
    caPerfil = (ca0 * (1 - Xres)) / (1 + epsilon * Xres)
    vPerfil = v0 * (1 + epsilon * Xres)

    # ==========================================
    # APRESENTAÇÃO DOS RESULTADOS NA INTERFACE
    # ==========================================

    st.title("Simulador de Reator Fluxo Pistão (PFR)")

    st.warning(
        "⚠️ **NOTE DE MODELAGEM:** Este simulador assume regime permanente em um reator **Isotérmico** e **Isobárico**."
    )

    st.info(
        f"Fase de Escoamento: **{fasesistema}** | "
        f"Constante Cinética: **{kfinal:.4e}** | "
        f"Fator de Expansão ($\epsilon$): **{epsilon:.4f}**"
    )

    # Cards atualizados dinamicamente
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversao de Saida (X)", f"{Xres[-1] * 100:.2f} %")
    col2.metric("Conc. Final Limitante (Ca)", f"{caPerfil[-1]:.3f} mol/L")
    col3.metric("Vazao Volumétrica de Saida", f"{vPerfil[-1]:.2f} L/min")
    col4.metric("Volume do Reator Usado (V)", f"{vtotalUsado:.2f} L")

    st.write("---")
    st.subheader("Gráficos Principais de Desempenho")

    colg1, colg2 = st.columns(2)

    with colg1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(vspan, Xres * 100, color='crimson', linewidth=2.5)
        ax1.set_title("Evolucao da Conversao ao Longo do Reator", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Volume do Reator (L)")
        ax1.set_ylabel("Conversao X (%)")
        ax1.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig1)

    with colg2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(vspan, caPerfil, color='royalblue', linewidth=2.5)
        ax2.set_title("Perfil de Concentracao do Limitante (Ca)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Volume do Reator (L)")
        ax2.set_ylabel("Concentracao (mol/L)")
        ax2.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig2)

    # Tabela rica com amostragem numerica
    if mostrartabela:
        st.write("---")
        st.subheader("Amostragem Numerica do Perfil Axial")
        
        dfresultados = pd.DataFrame({
            "Volume (L)": vspan,
            "Conversao (X)": Xres,
            "Concentracao A (mol/L)": caPerfil,
            "Vazão Local (L/min)": vPerfil
        })
        
        dfresumido = dfresultados.iloc[::10, :].copy()
        dfresumido = pd.concat([dfresumido, dfresultados.iloc[[-1]]]).drop_duplicates()
        
        st.dataframe(dfresumido.style.format({
            "Volume (L)": "{:.1f}",
            "Conversao (X)": "{:.4f}",
            "Concentracao A (mol/L)": "{:.4f}",
            "Vazão Local (L/min)": "{:.2f}"
        }), use_container_width=True)

    if simular:
        st.success("✅ Malha computacional resolvida com sucesso!")
