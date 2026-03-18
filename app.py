# app.py — ENGESTAT v1.1.0
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config import Config, OpcoesEntrada

from core.distribuicoes import (
    ajustar_distribuicoes, dados_para_pdf,
    dados_para_qqplot, dados_para_cdf, melhor_distribuicao
)
from core.descritiva import calcular_estatisticas
from core.normalidade import (
    shapiro_wilk, kolmogorov_smirnov_lilliefors,
    chi_square_goodness_of_fit, qq_plot_data
)
from core.estimacao import ic_media, ic_desvio, ic_variancia
from core.comparacao_2amostras import comparacao_completa
from core.anova import anova_um_fator, tukey_hsd

from utils.validacao import limpar_dados, validar_tamanho, detectar_outliers_iqr
from utils.interpretacao import (
    gerar_relatorio_completo, interpretar_normalidade, recomendar_teste,
    interpretar_ic_media, interpretar_ic_desvio, interpretar_ic_variancia,
    interpretar_teste_f, interpretar_teste_t, interpretar_anova, interpretar_tukey
)

# ── Logo SVG ──────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 70" width="210" height="68">
  <rect x="2" y="4" width="38" height="58" rx="6" ry="6"
        fill="none" stroke="#FFFFFF" stroke-width="2.4"/>
  <rect x="7" y="10" width="28" height="16" rx="2" ry="2"
        fill="rgba(255,255,255,0.08)" stroke="#FFFFFF" stroke-width="1.8"/>
  <path d="M10 26 Q14 12 21 12 Q28 12 32 26"
        fill="none" stroke="#60A5FA" stroke-width="2.2"/>
  <circle cx="11" cy="37" r="2.4" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="21" cy="37" r="2.4" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="31" cy="37" r="2.4" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="11" cy="46" r="2.4" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="21" cy="46" r="2.8" fill="#60A5FA"/>
  <circle cx="31" cy="46" r="2.4" fill="#FFFFFF" opacity="0.7"/>
  <circle cx="11" cy="55" r="2.4" fill="#FFFFFF" opacity="0.7"/>
  <rect x="17.5" y="52" width="15" height="5.5" rx="2.7" fill="#FFFFFF" opacity="0.7"/>
  <text x="52" y="38" font-family="'DM Sans',sans-serif" font-weight="700"
        font-size="25" fill="#FFFFFF" letter-spacing="-0.5">ENGE</text>
  <text x="116" y="38" font-family="'DM Sans',sans-serif" font-weight="300"
        font-size="25" fill="#60A5FA" letter-spacing="-0.5">STAT</text>
  <text x="52" y="54" font-family="'DM Sans',sans-serif" font-weight="400"
        font-size="9" fill="#93C5FD" letter-spacing="3">ANÁLISE ESTATÍSTICA</text>
</svg>
"""

# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #F0F4F8 !important;
    color: #1E2A3A !important;
}

/* ── Sidebar base ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0B1D3A 0%, #162E58 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 28px rgba(11,29,58,0.25) !important;
}

/* Texto branco em toda a sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
}

.sidebar-logo-area {
    padding: 28px 20px 22px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 6px;
}

/* Label do radio */
section[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #93C5FD !important;
}

/* Itens do radio */
section[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
section[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    margin: 0 8px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    font-size: 0.87rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    color: #E2EEFF !important;
    border: 1px solid transparent !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stRadio [role="radio"] { display: none !important; }

/* Status badges */
section[data-testid="stSidebar"] .stSuccess {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-left: 3px solid #10B981 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stSuccess,
section[data-testid="stSidebar"] .stSuccess * { color: #6EE7B7 !important; }

section[data-testid="stSidebar"] .stInfo {
    background: rgba(96,165,250,0.1) !important;
    border: 1px solid rgba(96,165,250,0.2) !important;
    border-left: 3px solid #60A5FA !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stInfo,
section[data-testid="stSidebar"] .stInfo * { color: #BAD9FF !important; }

/* Heading Desenvolvedores */
section[data-testid="stSidebar"] h3 {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #93C5FD !important;
}

/* Card desenvolvedores */
.sidebar-creators {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    line-height: 1.9 !important;
    font-size: 0.83rem !important;
}
.sidebar-creators, .sidebar-creators * { color: #E2EEFF !important; }
.sidebar-creators strong { color: #FFFFFF !important; font-weight: 500 !important; }

/* Caption / versão */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #6B93C4 !important;
    font-size: 0.72rem !important;
}

/* Botão resetar */
section[data-testid="stSidebar"] .stButton button {
    background: rgba(239,68,68,0.12) !important;
    color: #FCA5A5 !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 8px 16px !important; border-radius: 8px !important;
    box-shadow: none !important; transform: none !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(239,68,68,0.22) !important;
    transform: none !important; box-shadow: none !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 10px 16px !important;
}

/* ── Main area ── */
.main .block-container {
    padding: 2rem 2.8rem 3rem !important;
    max-width: 1200px !important;
}

.subtitle {
    font-size: 0.92rem !important; font-weight: 400 !important;
    color: #64748B !important; margin: 0 0 1.2rem 0 !important;
    border-left: 3px solid #1A3464 !important; padding-left: 12px !important;
}

h1 { font-size: 1.5rem !important; font-weight: 700 !important; color: #0B1D3A !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important; color: #1A3464 !important; }
h3 { font-size: 0.96rem !important; font-weight: 600 !important; color: #1A3464 !important; }

/* ── Welcome card ── */
.welcome-card {
    background: #FFFFFF !important; border-radius: 16px !important;
    padding: 3rem 2.5rem !important; text-align: center !important;
    box-shadow: 0 4px 24px rgba(11,29,58,0.07) !important;
    max-width: 680px !important; margin: 2rem auto !important;
    border: 1px solid #E2E8F0 !important;
}
.welcome-card .big-icon { font-size: 4rem; margin-bottom: 1rem; }
.welcome-card h2 { color: #0B1D3A !important; font-weight: 600 !important; margin-bottom: 0.5rem !important; }
.welcome-card p { color: #475569 !important; font-size: 1rem !important; max-width: 480px; margin: 0 auto 1.5rem; }

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important; padding: 18px 20px !important;
    box-shadow: 0 1px 3px rgba(11,29,58,0.05) !important;
    transition: box-shadow 0.2s ease !important;
}
div[data-testid="metric-container"]:hover { box-shadow: 0 4px 14px rgba(11,29,58,0.09) !important; }
div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.65rem !important; font-weight: 500 !important; color: #0B1D3A !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; font-weight: 600 !important;
    color: #64748B !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}

/* ── Buttons (main) ── */
.main .stButton button {
    background: #0B1D3A !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 22px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.87rem !important; font-weight: 500 !important;
    transition: all 0.16s ease !important;
    box-shadow: 0 2px 6px rgba(11,29,58,0.18) !important;
}
.main .stButton button:hover {
    background: #1A3464 !important;
    box-shadow: 0 4px 14px rgba(11,29,58,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background: #FFFFFF !important; border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.87rem !important; color: #1E2A3A !important;
    transition: border-color 0.16s ease, box-shadow 0.16s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: #1A3464 !important; box-shadow: 0 0 0 3px rgba(26,52,100,0.1) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important; border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #1A3464 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important; font-weight: 500 !important;
    color: #1E2A3A !important; font-size: 0.88rem !important; padding: 12px 16px !important;
}
.streamlit-expanderContent {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important; overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(11,29,58,0.06) !important; border: 1px solid #E2E8F0 !important;
}

/* ── Alerts (main) ── */
.stSuccess { background:#F0FDF4!important; border:1px solid #86EFAC!important; border-left:4px solid #22C55E!important; border-radius:8px!important; color:#166534!important; }
.stWarning { background:#FFFBEB!important; border:1px solid #FDE68A!important; border-left:4px solid #F59E0B!important; border-radius:8px!important; color:#92400E!important; }
.stError   { background:#FFF1F2!important; border:1px solid #FECDD3!important; border-left:4px solid #F43F5E!important; border-radius:8px!important; color:#881337!important; }
.stInfo    { background:#EFF6FF!important; border:1px solid #BFDBFE!important; border-left:4px solid #3B82F6!important; border-radius:8px!important; color:#1E40AF!important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 2px solid #E2E8F0 !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important; font-size: 0.87rem !important;
    font-weight: 500 !important; color: #64748B !important;
    padding: 10px 20px !important; background: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; margin-bottom: -2px !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #0B1D3A !important; background: transparent !important; }
.stTabs [aria-selected="true"] {
    color: #0B1D3A !important; border-bottom: 2px solid #0B1D3A !important;
    font-weight: 600 !important; background: transparent !important;
}

hr { border: none !important; border-top: 1px solid #E2E8F0 !important; margin: 22px 0 !important; }

.js-plotly-plot {
    border-radius: 12px !important; overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(11,29,58,0.06) !important; border: 1px solid #E2E8F0 !important;
}

.footer {
    text-align: center !important; margin-top: 3rem !important;
    padding: 1.2rem !important; color: #94A3B8 !important;
    font-size: 0.82rem !important; border-top: 1px solid #E2E8F0 !important;
}
.footer .creators { font-weight: 600 !important; color: #1A3464 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F0F4F8; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

.stCaption, small { color: #94A3B8 !important; font-size: 0.73rem !important; }
</style>
"""

PLOTLY_THEME = dict(
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font=dict(family="DM Sans", color="#1E2A3A", size=12),
    title_font=dict(size=14, color="#0B1D3A"),
    xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
)


class EngestatApp:

    def __init__(self):
        st.set_page_config(
            page_title="ENGESTAT",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._init_session()
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    def _init_session(self):
        if "dados" not in st.session_state:
            st.session_state.dados = None
        if "dados_validos" not in st.session_state:
            st.session_state.dados_validos = False
        if "modulo_atual" not in st.session_state:
            st.session_state.modulo_atual = "Inserção"

    def _exibir_boas_vindas(self):
        st.markdown("""
        <div class="welcome-card">
            <div class="big-icon">📊</div>
            <h2>Bem-vindo ao ENGESTAT</h2>
            <p>Ferramenta estatística desenvolvida para engenheiros.<br>
            Selecione um módulo no menu lateral e comece a analisar seus dados.</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 📊 Descritiva")
            st.caption("Média, mediana, desvio, CV")
        with col2:
            st.markdown("##### 📈 Normalidade")
            st.caption("Shapiro-Wilk, KS, Qui²")
        with col3:
            st.markdown("##### ⚖ Comparações")
            st.caption("Teste t, ANOVA, Tukey")
        st.markdown("---")

    # =============================
    # ENTRADAS
    # =============================

    def _input_manual(self):
        entrada = st.text_area(
            "📝 Insira valores separados por vírgula",
            placeholder="Exemplo: 25, 30, 28, 31, 29", height=100,
            help="Use vírgula ou ponto como separador decimal."
        )
        if entrada:
            dados_brutos = [x for x in entrada.replace(";", ",").replace(" ", "").split(",") if x]
            if dados_brutos:
                return limpar_dados(dados_brutos)
            st.warning("Nenhum valor válido encontrado.")
        return None

    def _input_upload(self):
        arquivo = st.file_uploader("📂 Envie CSV ou Excel", type=["csv", "xlsx", "xls"],
                                    help="Primeira coluna numérica será usada")
        if arquivo:
            try:
                df = pd.read_csv(arquivo) if arquivo.name.endswith(".csv") else pd.read_excel(arquivo)
                col_numericas = df.select_dtypes(include=["float64", "int64"])
                if col_numericas.empty:
                    st.warning("Nenhuma coluna numérica encontrada.")
                    return None
                dados = col_numericas.iloc[:, 0].dropna().tolist()
                return dados if dados else None
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {str(e)}")
        return None

    # =============================
    # GRÁFICOS
    # =============================

    def _graficos(self, dados):
        df = pd.DataFrame({"Valores": dados})
        n_bins = min(30, max(10, len(dados) // 5))
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Valores", nbins=n_bins, title="Histograma",
                               color_discrete_sequence=["#1A3464"], opacity=0.85)
            fig.update_layout(**PLOTLY_THEME, showlegend=False,
                              yaxis_title="Frequência", xaxis_title="Valores")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(df, y="Valores", title="Boxplot",
                          color_discrete_sequence=["#1A3464"], points="all")
            fig2.update_traces(boxmean=True,
                               marker=dict(size=5, color="#3B82F6", opacity=0.75,
                                           line=dict(width=1, color="#1A3464")),
                               line=dict(width=2), fillcolor="rgba(26,52,100,0.18)")
            fig2.update_layout(**PLOTLY_THEME, showlegend=False, yaxis_title="Valores")
            st.plotly_chart(fig2, use_container_width=True)

    # =============================
    # MÓDULOS
    # =============================

    def _modulo_insercao(self):
        st.header("📥 Inserção de Dados")
        st.markdown("Insira seus dados de uma das formas abaixo para iniciar a análise.")
        opcao = st.radio("Escolha a fonte dos dados:",
                         [OpcoesEntrada.MANUAL.value, OpcoesEntrada.UPLOAD.value], horizontal=True)
        dados = self._input_manual() if opcao == OpcoesEntrada.MANUAL.value else self._input_upload()
        if dados is not None:
            valido, erro = validar_tamanho(dados)
            if not valido:
                st.error(f"❌ {erro}")
                return
            st.session_state.dados = dados
            st.session_state.dados_validos = True
            st.success(f"✅ {len(dados)} valores carregados com sucesso!")
            with st.expander("👁️ Preview dos dados"):
                st.dataframe(pd.DataFrame(dados, columns=["Valores"]).describe(),
                             use_container_width=True)
            if st.button("🚀 Analisar dados", type="primary", use_container_width=True):
                st.session_state.modulo_atual = "Estatística Descritiva"
                st.rerun()

    def _modulo_descritiva(self):
        st.header("📊 Estatística Descritiva")
        col_info1, col_info2 = st.columns([3, 1])
        with col_info1:
            st.caption(f"📋 Analisando {len(st.session_state.dados)} valores")
        with col_info2:
            if st.button("🔄 Novos dados"):
                st.session_state.dados = None
                st.session_state.dados_validos = False
                st.session_state.modulo_atual = "Inserção"
                st.rerun()
        dados = st.session_state.dados
        with st.spinner("Processando estatísticas..."):
            est = calcular_estatisticas(dados)
            _, outliers = detectar_outliers_iqr(dados)
            percentual = (sum(outliers) / len(outliers) * 100) if outliers else 0
        st.subheader("📌 Métricas Principais")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Média", f"{est.media:.2f}")
        with col2: st.metric("Desvio Padrão", f"{est.desvio_padrao:.2f}")
        with col3:
            cv_value = f"{est.coeficiente_variacao:.1f}%" if est.coeficiente_variacao else "N/A"
            st.metric("CV", cv_value)
        with col4: st.metric("Outliers", f"{percentual:.1f}%")
        st.subheader("📈 Visualizações")
        self._graficos(dados)
        with st.expander("📋 Ver relatório técnico completo", expanded=True):
            st.markdown(gerar_relatorio_completo(est, percentual))

    def _modulo_normalidade(self):
        st.header("📈 Testes de Normalidade")
        dados = st.session_state.dados
        with st.expander("ℹ️ Sobre os testes", expanded=False):
            st.markdown("""
            - **Shapiro-Wilk**: recomendado para amostras pequenas (n < 50)
            - **Kolmogorov-Smirnov (Lilliefors)**: para amostras maiores
            - **Qui-Quadrado**: teste de aderência, sensível ao número de classes
            """)
        col1, _ = st.columns([2, 1])
        with col1:
            n_classes = st.slider("Número de classes (Qui-Quadrado)", min_value=4, max_value=15, value=8)
        with st.spinner("Executando testes..."):
            try:
                res_shapiro = shapiro_wilk(dados)
                res_ks = kolmogorov_smirnov_lilliefors(dados)
                res_chi2 = chi_square_goodness_of_fit(dados, n_classes)
            except Exception as e:
                st.error(f"Erro ao executar testes: {e}")
                return
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🧪 Shapiro-Wilk")
            st.metric("W", f"{res_shapiro['estatistica']:.4f}")
            st.metric("p-valor", f"{res_shapiro['pvalor']:.4f}")
            st.info(interpretar_normalidade(res_shapiro['pvalor']))
        with col2:
            st.subheader("📊 Kolmogorov-Smirnov")
            st.metric("D", f"{res_ks['estatistica']:.4f}")
            st.metric("p-valor", f"{res_ks['pvalor']:.4f}")
            st.info(interpretar_normalidade(res_ks['pvalor']))
        with col3:
            st.subheader("📐 Qui-Quadrado")
            st.metric("χ²", f"{res_chi2['estatistica']:.4f}")
            st.metric("p-valor", f"{res_chi2['pvalor']:.4f}")
            st.metric("gl", f"{res_chi2['df']}")
            st.info(interpretar_normalidade(res_chi2['pvalor']))
        st.divider()
        st.subheader("📉 Q-Q Plot")
        quantis_teoricos, dados_ordenados = qq_plot_data(dados)
        df_qq = pd.DataFrame({'Quantis teóricos': quantis_teoricos, 'Dados ordenados': dados_ordenados})
        fig_qq = px.scatter(df_qq, x='Quantis teóricos', y='Dados ordenados',
                            title='Q-Q Plot — Comparação com Distribuição Normal',
                            trendline='ols', color_discrete_sequence=['#1A3464'])
        max_val = max(abs(quantis_teoricos.max()), abs(dados_ordenados.max())) * 1.1
        fig_qq.add_shape(type='line', x0=-max_val, y0=-max_val, x1=max_val, y1=max_val,
                         line=dict(color='#EF4444', dash='dash', width=2))
        fig_qq.update_layout(**PLOTLY_THEME)
        fig_qq.update_xaxes(range=[-max_val, max_val], showgrid=True, gridcolor="#F1F5F9")
        fig_qq.update_yaxes(range=[-max_val, max_val], showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_qq, use_container_width=True)
        st.divider()
        st.subheader("🔍 Recomendação")
        count_normal = sum([res_shapiro['normal'], res_ks['normal'], res_chi2['normal']])
        if count_normal >= 2:
            st.success(recomendar_teste(True))
        else:
            st.warning(recomendar_teste(False))

    def _modulo_distribuicoes(self):
        st.header("📈 Comparação de Distribuições")
        dados = st.session_state.dados
        with st.spinner("Ajustando distribuições..."):
            resultados = ajustar_distribuicoes(dados)
        st.subheader("📋 Tabela de Ajuste")
        df_result = pd.DataFrame([{
            "Distribuição": r["Distribuição"], "AIC": round(r["AIC"], 2),
            "BIC": round(r["BIC"], 2), "KS p-valor": round(r["KS_p"], 4),
            "AD Estat.": round(r["AD_Stat"], 4) if r["AD_Stat"] else "N/A"
        } for r in resultados])
        st.dataframe(df_result, use_container_width=True, hide_index=True)
        melhor = melhor_distribuicao(resultados, criterio="AIC")
        st.success(f"✅ **Melhor distribuição (menor AIC):** {melhor}")
        tab1, tab2, tab3 = st.tabs(["📊 PDF + Histograma", "📉 Q-Q Plots", "📈 CDF Empírica vs Teórica"])
        cores = px.colors.qualitative.Plotly
        with tab1:
            dados_plot = dados_para_pdf(dados, resultados)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=dados, name="Dados", histnorm='probability density',
                                        nbinsx=min(30, len(dados)//5), marker_color='#CBD5E1', opacity=0.65))
            for i, pdf in enumerate(dados_plot["pdfs"]):
                fig.add_trace(go.Scatter(x=dados_plot["x_vals"], y=pdf["y"], mode='lines',
                                          name=pdf["nome"], line=dict(color=cores[i % len(cores)], width=2)))
            fig.update_layout(**PLOTLY_THEME, xaxis_title="Valor", yaxis_title="Densidade",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            qq_data = dados_para_qqplot(dados, resultados)
            cols = st.columns(min(2, len(qq_data)))
            for i, qq in enumerate(qq_data):
                with cols[i % 2]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=qq["teorico"], y=qq["observado"],
                                              mode='markers', marker=dict(color='#1A3464', size=4)))
                    mn = min(min(qq["teorico"]), min(qq["observado"]))
                    mx = max(max(qq["teorico"]), max(qq["observado"]))
                    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                              line=dict(color='#EF4444', dash='dash')))
                    fig.update_layout(**PLOTLY_THEME, title=qq['nome'],
                                       xaxis_title="Quantis teóricos", yaxis_title="Quantis observados",
                                       width=400, height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        with tab3:
            cdf_data = dados_para_cdf(dados, resultados)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cdf_data[0]["x"], y=cdf_data[0]["cdf_emp"],
                                      mode='lines+markers', name='CDF Empírica',
                                      line=dict(color='#0B1D3A', width=2), marker=dict(size=2)))
            for i, cdf in enumerate(cdf_data):
                fig.add_trace(go.Scatter(x=cdf["x"], y=cdf["cdf_teo"], mode='lines',
                                          name=cdf["nome"], line=dict(color=cores[i % len(cores)], width=2)))
            fig.update_layout(**PLOTLY_THEME, xaxis_title="Valor", yaxis_title="Probabilidade acumulada",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

    def _modulo_estimacao(self):
        st.header("📐 Estimação por Intervalos de Confiança")
        dados = st.session_state.dados
        if dados is None:
            st.warning("Este módulo requer dados. Por favor, insira dados primeiro.")
            return
        alpha = st.slider("Nível de significância α (1 − confiança)", 0.01, 0.20, 0.05, 0.01)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Média")
            sigma_conhecido = st.number_input("σ conhecido (0 = desconhecido)", value=0.0, step=0.1)
            res_media = (ic_media(np.array(dados), alpha=alpha, sigma_conhecido=sigma_conhecido)
                         if sigma_conhecido != 0 else ic_media(np.array(dados), alpha=alpha))
            st.metric("Média amostral", f"{res_media['media']:.3f}")
            st.metric("IC", f"[{res_media['li']:.3f}, {res_media['ls']:.3f}]")
            st.info(interpretar_ic_media(res_media))
        with col2:
            st.subheader("Desvio Padrão")
            res_desvio = ic_desvio(np.array(dados), alpha=alpha)
            st.metric("Desvio amostral S", f"{res_desvio['S']:.3f}")
            st.metric("IC", f"[{res_desvio['li']:.3f}, {res_desvio['ls']:.3f}]")
            if 'aprox_normal' in res_desvio:
                an = res_desvio['aprox_normal']
                st.metric("IC aprox. normal", f"[{an['li']:.3f}, {an['ls']:.3f}]")
            st.info(interpretar_ic_desvio(res_desvio))
        st.divider()
        st.subheader("Variância")
        res_var = ic_variancia(np.array(dados), alpha=alpha)
        col1, col2 = st.columns(2)
        col1.metric("Variância amostral S²", f"{res_var['S2']:.3f}")
        col2.metric("IC", f"[{res_var['li']:.3f}, {res_var['ls']:.3f}]")
        st.info(interpretar_ic_variancia(res_var))

    def _modulo_comparacao_2grupos(self):
        st.header("⚖ Comparação de Duas Amostras Independentes")
        st.markdown("Defina os dois grupos manualmente (valores separados por vírgula).")
        col1, col2 = st.columns(2)
        with col1:
            grupo1_input = st.text_area("Grupo 1", placeholder="Ex: 10, 12, 15", key="g1")
        with col2:
            grupo2_input = st.text_area("Grupo 2", placeholder="Ex: 9, 11, 14", key="g2")
        if not grupo1_input or not grupo2_input:
            st.info("Digite os dois grupos para iniciar a análise.")
            return
        grupo1 = limpar_dados([x.strip() for x in grupo1_input.replace(';', ',').split(',') if x.strip()])
        grupo2 = limpar_dados([x.strip() for x in grupo2_input.replace(';', ',').split(',') if x.strip()])
        if grupo1 is None or grupo2 is None or len(grupo1) < 2 or len(grupo2) < 2:
            st.error("Cada grupo deve ter pelo menos dois números válidos.")
            return
        alpha = st.slider("Nível de significância α", 0.01, 0.20, 0.05, 0.01)
        with st.spinner("Executando testes..."):
            res = comparacao_completa(np.array(grupo1), np.array(grupo2), alpha)
        st.subheader("Resultados")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Grupo 1**")
            st.write(f"n = {len(grupo1)}, média = {np.mean(grupo1):.3f}, desvio = {np.std(grupo1, ddof=1):.3f}")
        with colB:
            st.markdown("**Grupo 2**")
            st.write(f"n = {len(grupo2)}, média = {np.mean(grupo2):.3f}, desvio = {np.std(grupo2, ddof=1):.3f}")
        st.divider()
        st.info(interpretar_teste_f(res['teste_f']))
        st.info(interpretar_teste_t(res['teste_t']))

    def _modulo_anova_tukey(self):
        st.header("📊 ANOVA e Teste de Tukey")
        st.markdown("Insira os dados para cada grupo. Você pode adicionar quantos grupos quiser.")
        if "num_grupos" not in st.session_state:
            st.session_state.num_grupos = 3
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➕ Adicionar grupo"):
                st.session_state.num_grupos += 1
        with col2:
            if st.button("➖ Remover último grupo") and st.session_state.num_grupos > 2:
                st.session_state.num_grupos -= 1
        grupos_dict = {}
        for i in range(st.session_state.num_grupos):
            nome = st.text_input(f"Nome do Grupo {i+1}", value=f"Grupo {i+1}", key=f"nome_{i}")
            dados_input = st.text_area(f"Dados do {nome}", key=f"dados_{i}", height=80)
            if dados_input:
                dados = limpar_dados([x.strip() for x in dados_input.replace(';', ',').split(',') if x.strip()])
                if dados and len(dados) >= 2:
                    grupos_dict[nome] = dados
                else:
                    st.warning(f"Grupo {nome} inválido ou com menos de 2 valores.")
        if len(grupos_dict) < 2:
            st.info("Adicione pelo menos dois grupos válidos para realizar ANOVA.")
            return
        alpha = st.slider("Nível de significância α", 0.01, 0.20, 0.05, 0.01)
        with st.spinner("Calculando ANOVA..."):
            res_anova = anova_um_fator(grupos_dict, alpha)
        st.subheader("Quadro da ANOVA")
        st.dataframe(res_anova['tabela'].round(4), hide_index=True, use_container_width=True)
        st.info(interpretar_anova(res_anova))
        if res_anova['rejeita_h0']:
            st.divider()
            st.subheader("Teste de Tukey (Comparações Múltiplas)")
            with st.spinner("Calculando Tukey HSD..."):
                df_tukey = tukey_hsd(grupos_dict, alpha)
            st.dataframe(df_tukey, hide_index=True, use_container_width=True)
            st.info(interpretar_tukey(df_tukey))

    # =============================
    # RUN
    # =============================

    def run(self):
        # Apenas o subtítulo — sem header duplicado com capacete
        st.markdown('<p class="subtitle">Ferramenta Estatística para Engenharias</p>',
                    unsafe_allow_html=True)

        with st.sidebar:
            # Logo maior
            st.markdown(f'<div class="sidebar-logo-area">{LOGO_SVG}</div>',
                        unsafe_allow_html=True)

            if st.session_state.dados_validos:
                st.success(f"✅ Dados ativos: {len(st.session_state.dados)} valores")
            else:
                st.info("📥 Nenhum dado carregado")

            st.divider()

            modulo_selecionado = st.radio(
                "Módulos:",
                ["Inserção de Dados", "Estatística Descritiva", "Testes de Normalidade",
                 "Comparação de Distribuições", "Estimação de Parâmetros",
                 "Comparação de 2 Grupos", "ANOVA e Tukey"],
                key="sidebar_radio"
            )

            mapa_modulos = {
                "Inserção de Dados": "Inserção",
                "Estatística Descritiva": "Estatística Descritiva",
                "Testes de Normalidade": "Testes de Normalidade",
                "Comparação de Distribuições": "Comparação de Distribuições",
                "Estimação de Parâmetros": "Estimação de Parâmetros",
                "Comparação de 2 Grupos": "Comparação de 2 Grupos",
                "ANOVA e Tukey": "ANOVA e Tukey"
            }
            st.session_state.modulo_atual = mapa_modulos[modulo_selecionado]

            st.divider()
            st.markdown("### 👥 Desenvolvedores")
            st.markdown("""
            <div class="sidebar-creators">
                <strong>Alverlando Silva Ricardo</strong><br>
                <strong>Jadson Lourenço de Sousa Santana</strong><br>
                <strong>Lara Gabriela Lisboa de Souza</strong>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            st.caption("🔧 Versão 1.1.0")
            st.caption("📚 Projeto de Estatística para Engenharias")

            if st.session_state.dados_validos:
                st.divider()
                if st.button("🔄 Resetar dados", use_container_width=True):
                    st.session_state.dados = None
                    st.session_state.dados_validos = False
                    st.session_state.modulo_atual = "Inserção"
                    st.rerun()

        # Roteamento
        modulo = st.session_state.modulo_atual
        requer_dados = modulo in ["Estatística Descritiva", "Testes de Normalidade",
                                   "Comparação de Distribuições", "Estimação de Parâmetros"]

        if modulo == "Inserção":
            self._modulo_insercao()
        elif requer_dados and not st.session_state.dados_validos:
            st.warning("Este módulo requer dados. Por favor, insira dados primeiro.")
            self._exibir_boas_vindas()
        elif modulo == "Estatística Descritiva":
            self._modulo_descritiva()
        elif modulo == "Testes de Normalidade":
            self._modulo_normalidade()
        elif modulo == "Comparação de Distribuições":
            self._modulo_distribuicoes()
        elif modulo == "Estimação de Parâmetros":
            self._modulo_estimacao()
        elif modulo == "Comparação de 2 Grupos":
            self._modulo_comparacao_2grupos()
        elif modulo == "ANOVA e Tukey":
            self._modulo_anova_tukey()

        st.markdown("""
        <div class="footer">
            <span class="creators">ENGESTAT</span> — desenvolvido por
            <strong>Alverlando Silva Ricardo</strong>,
            <strong>Jadson Lourenço de Sousa Santana</strong> e
            <strong>Lara Gabriela Lisboa de Souza</strong>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = EngestatApp()
    app.run()