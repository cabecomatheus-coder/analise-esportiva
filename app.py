# -*- coding: utf-8 -*-
"""
app.py
==============================================================================
SILVER BOOL — DASHBOARD DE ANÁLISE PREDITIVA DE APOSTAS ESPORTIVAS
==============================================================================
"""

import os
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import poisson

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS CUSTOMIZADO
# ==============================================================================

st.set_page_config(
    page_title="SILVER BOOL",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS Personalizada (Modern Dark Premium UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Ocultar elementos desnecessários do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    .stApp {
        background: radial-gradient(circle at 15% 0%, #121820 0%, #0d1117 45%, #0a0e13 100%);
    }

    /* ============ HERO ============ */
    .hero-container {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        padding: 28px 24px;
        border-radius: 18px;
        border: 1px solid #30363d;
        box-shadow: 0 12px 32px rgba(0,0,0,0.45);
        margin-bottom: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -60%; left: -20%;
        width: 60%; height: 220%;
        background: radial-gradient(circle, rgba(56,239,125,0.10) 0%, rgba(0,0,0,0) 70%);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38ef7d 0%, #11998e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        position: relative;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        font-weight: 500;
        position: relative;
    }

    /* ============ SUMMARY STRIP ============ */
    .summary-strip {
        display: flex;
        gap: 12px;
        margin-bottom: 22px;
        flex-wrap: wrap;
    }
    .summary-chip {
        flex: 1;
        min-width: 140px;
        background: linear-gradient(145deg, #161b22 0%, #12161c 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .summary-chip .num {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f0f6fc;
    }
    .summary-chip .lbl {
        font-size: 0.72rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }

    /* ============ LEAGUE HEADER ============ */
    .league-header {
        display: flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(90deg, rgba(56,239,125,0.10) 0%, rgba(22,27,34,0) 60%);
        border-left: 3px solid #38ef7d;
        padding: 10px 16px;
        border-radius: 8px;
        margin: 22px 0 12px 0;
    }
    .league-header .league-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .league-header .league-count {
        font-size: 0.75rem;
        color: #8b949e;
        font-weight: 600;
        margin-left: auto;
    }

    /* ============ CARDS DE ESTATÍSTICAS ============ */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        border-color: #38ef7d;
        transform: translateY(-2px);
    }
    .metric-card.favorite { border-color: #238636; background: linear-gradient(145deg, #161b22 0%, #10251a 100%); }
    .metric-card.underdog { border-color: #30363d; }

    .metric-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 6px;
        letter-spacing: 0.02em;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f0f6fc;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #38ef7d;
        font-weight: 600;
        margin-top: 4px;
    }

    /* ============ AVATAR DO TIME ============ */
    .team-block {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .team-avatar {
        width: 42px;
        height: 42px;
        min-width: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
        color: #0d1117;
    }
    .avatar-home { background: linear-gradient(135deg, #38ef7d, #11998e); }
    .avatar-away { background: linear-gradient(135deg, #FF9966, #FF5E62); }
    .team-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .team-role {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* ============ BADGES DE FORMA ============ */
    .badge {
        display: inline-block;
        width: 22px;
        height: 22px;
        line-height: 22px;
        border-radius: 50%;
        font-size: 0.7rem;
        font-weight: 800;
        margin: 0 2px;
        text-align: center;
    }
    .badge-v { background-color: #238636; color: #ffffff; }
    .badge-e { background-color: #9e6a03; color: #ffffff; }
    .badge-d { background-color: #da3633; color: #ffffff; }

    /* ============ STAT PILLS ============ */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 0.78rem;
        color: #c9d1d9;
        margin: 3px 4px 3px 0;
    }
    .stat-pill b { color: #f0f6fc; }

    /* ============ BARRA DE PROBABILIDADE 1X2 ============ */
    .prob-bar-wrap {
        width: 100%;
        height: 30px;
        border-radius: 8px;
        overflow: hidden;
        display: flex;
        margin: 10px 0 4px 0;
        border: 1px solid #30363d;
    }
    .prob-bar-seg {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
        color: #0d1117;
        transition: width 0.3s ease;
    }
    .seg-home { background: linear-gradient(90deg, #38ef7d, #2dd4bf); }
    .seg-draw { background: #F7B731; }
    .seg-away { background: linear-gradient(90deg, #FF9966, #FF5E62); }

    /* ============ LIQUIDEZ ============ */
    .liquidity-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        margin-left: 8px;
    }
    .liq-high { background: rgba(56,239,125,0.15); color: #38ef7d; border: 1px solid #238636; }
    .liq-mid { background: rgba(247,183,49,0.15); color: #F7B731; border: 1px solid #9e6a03; }
    .liq-low { background: rgba(139,148,158,0.15); color: #8b949e; border: 1px solid #30363d; }

    /* ============ EXPANDERS (CONFRONTOS) ============ */
    .streamlit-expanderHeader, div[data-testid="stExpander"] summary {
        background-color: #161b22 !important;
        border-radius: 10px !important;
        border: 1px solid #30363d !important;
        font-weight: 600 !important;
        color: #c9d1d9 !important;
    }
    div[data-testid="stExpander"] summary:hover {
        border-color: #38ef7d !important;
    }
    div[data-testid="stExpander"] {
        border: none !important;
        margin-bottom: 8px;
    }

    /* ============ TABS ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1c2530 !important;
        color: #38ef7d !important;
    }

    /* ============ TABELA CUSTOMIZADA ============ */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* ============ SIDEBAR ============ */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    .sidebar-footer {
        font-size: 0.72rem;
        color: #484f58;
        text-align: center;
        margin-top: 24px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

FUSO_BRASIL = ZoneInfo("America/Sao_Paulo")

# ==============================================================================
# 2. CAMADA DE DADOS
# ==============================================================================

@dataclass
class EstatisticasTime:
    nome: str
    gols_marcados_media: float
    gols_sofridos_media: float
    escanteios_favor_media: float
    escanteios_contra_media: float
    cartoes_amarelos_media: float
    cartoes_vermelhos_media: float
    finalizacoes_gol_media: float
    ultimos_5_jogos: List[str]
    forca_ataque: float = field(init=False)
    forca_defesa: float = field(init=False)

    def __post_init__(self):
        self.forca_ataque = self.gols_marcados_media
        self.forca_defesa = self.gols_sofridos_media


@dataclass
class Partida:
    id: str
    liga_nome: str
    mandante: str
    visitante: str
    horario: datetime
    liquidez: float
    estat_mandante: EstatisticasTime
    estat_visitante: EstatisticasTime


def _gerar_ultimos_jogos(seed: int) -> List[str]:
    rng = random.Random(seed)
    return rng.choices(["V", "E", "D"], weights=[0.45, 0.30, 0.25], k=5)


def _gerar_estatisticas_aleatorias(nome_time: str, seed: int) -> EstatisticasTime:
    rng = random.Random(seed)
    return EstatisticasTime(
        nome=nome_time,
        gols_marcados_media=round(rng.uniform(0.9, 2.4), 2),
        gols_sofridos_media=round(rng.uniform(0.6, 1.8), 2),
        escanteios_favor_media=round(rng.uniform(3.5, 7.5), 2),
        escanteios_contra_media=round(rng.uniform(3.0, 6.5), 2),
        cartoes_amarelos_media=round(rng.uniform(1.5, 3.2), 2),
        cartoes_vermelhos_media=round(rng.uniform(0.0, 0.25), 2),
        finalizacoes_gol_media=round(rng.uniform(3.0, 7.0), 2),
        ultimos_5_jogos=_gerar_ultimos_jogos(seed),
    )


def buscar_partidas_api_global(data: str, api_key: str) -> List[Partida]:
    headers = {
        "x-apisports-key": api_key,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    partidas = []
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"date": data, "timezone": "America/Sao_Paulo"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=12)
        if resp.status_code == 200:
            jogos = resp.json().get("response", [])
            for item in jogos:
                pais = str(item['league']['country']).strip()
                liga_nome_bruto = str(item['league']['name']).strip()

                if "russia" in pais.lower() or "russia" in liga_nome_bruto.lower():
                    continue

                f_id = str(item["fixture"]["id"])
                liga_nome = f"{liga_nome_bruto} ({pais})"
                mandante_nome = item["teams"]["home"]["name"]
                visitante_nome = item["teams"]["away"]["name"]

                date_raw = item["fixture"]["date"]
                horario_utc = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                horario_local = horario_utc.astimezone(FUSO_BRASIL)

                seed_m = abs(hash(mandante_nome)) % 10000
                seed_v = abs(hash(visitante_nome)) % 10000

                partidas.append(Partida(
                    id=f_id,
                    liga_nome=liga_nome,
                    mandante=mandante_nome,
                    visitante=visitante_nome,
                    horario=horario_local,
                    liquidez=round(random.Random(f_id).uniform(50_000, 3_000_000), 2),
                    estat_mandante=_gerar_estatisticas_aleatorias(mandante_nome, seed_m),
                    estat_visitante=_gerar_estatisticas_aleatorias(visitante_nome, seed_v),
                ))
    except Exception as e:
        st.error(f"Erro na conexão com a API: {e}")

    return partidas


_TIMES_MUNDO = [
    ("Premier League (Inglaterra)", "Manchester City", "Arsenal"),
    ("Premier League (Inglaterra)", "Liverpool", "Chelsea"),
    ("Brasileirão Série A (Brasil)", "Palmeiras", "Flamengo"),
    ("Brasileirão Série A (Brasil)", "Fluminense", "Botafogo"),
    ("La Liga (Espanha)", "Real Madrid", "Barcelona"),
    ("La Liga (Espanha)", "Atlético de Madrid", "Sevilla"),
    ("Serie A (Itália)", "Inter de Milão", "Juventus"),
    ("Bundesliga (Alemanha)", "Bayern de Munique", "Bayer Leverkusen"),
]


@st.cache_data(ttl=3600, show_spinner=False)
def simular_partidas_do_dia(data_ref: str) -> List[Partida]:
    partidas = []
    seed_base = int(pd.Timestamp(data_ref).strftime("%Y%m%d"))

    for i, (liga, mandante, visitante) in enumerate(_TIMES_MUNDO):
        rng = random.Random(seed_base + i)
        data_obj = datetime.strptime(data_ref, "%Y-%m-%d")
        horario = data_obj.replace(
            hour=rng.choice([11, 13, 15, 16, 18, 19, 21]),
            minute=rng.choice([0, 15, 30, 45]),
            tzinfo=FUSO_BRASIL
        )

        partidas.append(Partida(
            id=f"sim_{i}",
            liga_nome=liga,
            mandante=mandante,
            visitante=visitante,
            horario=horario,
            liquidez=round(rng.uniform(150_000, 4_500_000), 2),
            estat_mandante=_gerar_estatisticas_aleatorias(mandante, seed_base + i),
            estat_visitante=_gerar_estatisticas_aleatorias(visitante, seed_base + i + 500),
        ))
    return partidas


def obter_partidas_do_dia(data_ref: str, modo: str, api_key: str) -> List[Partida]:
    if modo == "API-Football":
        if not api_key:
            st.error("Insira sua API Key no menu lateral para carregar dados reais.")
            return []
        partidas_reais = buscar_partidas_api_global(data_ref, api_key)
        if not partidas_reais:
            st.warning("Nenhum jogo retornado pela API para esta data.")
            return []
        return partidas_reais
    return simular_partidas_do_dia(data_ref)

# ==============================================================================
# 3. MODELAGEM MATEMÁTICA
# ==============================================================================

MEDIA_GOLS_LIGA = 1.35
VANTAGEM_MANDANTE = 1.10


def calcular_lambda_gols(estat_mandante: EstatisticasTime, estat_visitante: EstatisticasTime) -> Tuple[float, float]:
    lm = (estat_mandante.forca_ataque / MEDIA_GOLS_LIGA) * (estat_visitante.forca_defesa / MEDIA_GOLS_LIGA) * MEDIA_GOLS_LIGA * VANTAGEM_MANDANTE
    lv = (estat_visitante.forca_ataque / MEDIA_GOLS_LIGA) * (estat_mandante.forca_defesa / MEDIA_GOLS_LIGA) * MEDIA_GOLS_LIGA
    return max(lm, 0.05), max(lv, 0.05)


def matriz_placares_poisson(lambda_mandante: float, lambda_visitante: float, max_gols: int = 8) -> np.ndarray:
    pm = [poisson.pmf(i, lambda_mandante) for i in range(max_gols + 1)]
    pv = [poisson.pmf(j, lambda_visitante) for j in range(max_gols + 1)]
    return np.outer(pm, pv)


def odd_justa(prob: float) -> float:
    return round(1 / prob, 2) if prob > 0 else float("inf")


def obter_top_10_placares(matriz: np.ndarray) -> List[Dict]:
    placares = []
    for i in range(min(matriz.shape[0], 6)):
        for j in range(min(matriz.shape[1], 6)):
            prob = matriz[i, j]
            placares.append({
                "placar": f"{i} x {j}",
                "probabilidade": prob,
                "prob_percentual": prob * 100,
                "odd_justa": odd_justa(prob)
            })
    return sorted(placares, key=lambda x: x["probabilidade"], reverse=True)[:10]


def probabilidades_1x2(matriz: np.ndarray) -> Dict[str, float]:
    return {
        "mandante": float(np.tril(matriz, -1).sum()),
        "empate": float(np.trace(matriz)),
        "visitante": float(np.triu(matriz, 1).sum())
    }


def probabilidade_over_under(matriz: np.ndarray, linha: float) -> Dict[str, float]:
    total_over = sum(matriz[i, j] for i in range(matriz.shape[0]) for j in range(matriz.shape[1]) if i + j > linha)
    return {"over": total_over, "under": 1.0 - total_over}


def calcular_mercado_escanteios(estat_m: EstatisticasTime, estat_v: EstatisticasTime) -> dict:
    l_m = (estat_m.escanteios_favor_media + estat_v.escanteios_contra_media) / 2
    l_v = (estat_v.escanteios_favor_media + estat_m.escanteios_contra_media) / 2
    matriz_cantos = matriz_placares_poisson(l_m, l_v, max_gols=15)
    return {
        "lambda_mandante": round(l_m, 2),
        "lambda_visitante": round(l_v, 2),
        "total_esperado": round(l_m + l_v, 2),
        "over_under": {linha: probabilidade_over_under(matriz_cantos, linha) for linha in [8.5, 9.5, 10.5]}
    }


def analisar_partida_completa(partida: Partida) -> dict:
    lm, lv = calcular_lambda_gols(partida.estat_mandante, partida.estat_visitante)
    matriz = matriz_placares_poisson(lm, lv)
    return {
        "lambda_mandante": round(lm, 2),
        "lambda_visitante": round(lv, 2),
        "matriz_placares": matriz,
        "top_10_placares": obter_top_10_placares(matriz),
        "probs_1x2": probabilidades_1x2(matriz),
        "over_under_gols": {linha: probabilidade_over_under(matriz, linha) for linha in [1.5, 2.5, 3.5]},
        "escanteios": calcular_mercado_escanteios(partida.estat_mandante, partida.estat_visitante),
        "secundarios": {
            "cartoes_amarelos": round(partida.estat_mandante.cartoes_amarelos_media + partida.estat_visitante.cartoes_amarelos_media, 2),
            "cartoes_vermelhos": round(partida.estat_mandante.cartoes_vermelhos_media + partida.estat_visitante.cartoes_vermelhos_media, 2),
            "finalizacoes": round(partida.estat_mandante.finalizacoes_gol_media + partida.estat_visitante.finalizacoes_gol_media, 2),
        }
    }

# ==============================================================================
# 4. COMPONENTES VISUAIS (INTERFACE GRÁFICA)
# ==============================================================================

def _iniciais(nome: str) -> str:
    partes = [p for p in nome.replace("-", " ").split(" ") if p]
    if len(partes) == 1:
        return partes[0][:2].upper()
    return (partes[0][0] + partes[-1][0]).upper()


def render_badge_forma(jogos: List[str]) -> str:
    html = ""
    for r in jogos:
        cls = "badge-v" if r == "V" else ("badge-e" if r == "E" else "badge-d")
        html += f'<span class="badge {cls}">{r}</span>'
    return html


def render_team_header(nome: str, papel: str) -> str:
    cls_avatar = "avatar-home" if papel == "MANDANTE" else "avatar-away"
    return f"""
    <div class="team-block">
        <div class="team-avatar {cls_avatar}">{_iniciais(nome)}</div>
        <div>
            <div class="team-role">{papel}</div>
            <div class="team-name">{nome}</div>
        </div>
    </div>
    """


def render_stat_pills(estat: EstatisticasTime) -> str:
    return f"""
    <span class="stat-pill">⚽ Marc <b>{estat.gols_marcados_media}</b></span>
    <span class="stat-pill">🛡️ Sofr <b>{estat.gols_sofridos_media}</b></span>
    <span class="stat-pill">🚩 Cantos <b>{estat.escanteios_favor_media}</b></span>
    <span class="stat-pill">🟨 Cartões <b>{estat.cartoes_amarelos_media}</b></span>
    <span class="stat-pill">🎯 Finaliz. <b>{estat.finalizacoes_gol_media}</b></span>
    """


def card_metrica_html(label: str, valor: str, sub: str = "", destaque: bool = False) -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ''
    cls = "metric-card favorite" if destaque else "metric-card"
    return f"""
    <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{valor}</div>
        {sub_html}
    </div>
    """


def render_prob_stacked_bar(probs: Dict[str, float], mandante: str, visitante: str) -> str:
    hm = max(probs["mandante"] * 100, 0.5)
    he = max(probs["empate"] * 100, 0.5)
    ha = max(probs["visitante"] * 100, 0.5)
    return f"""
    <div class="prob-bar-wrap">
        <div class="prob-bar-seg seg-home" style="width:{hm}%;">{probs['mandante']*100:.0f}%</div>
        <div class="prob-bar-seg seg-draw" style="width:{he}%;">{probs['empate']*100:.0f}%</div>
        <div class="prob-bar-seg seg-away" style="width:{ha}%;">{probs['visitante']*100:.0f}%</div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#8b949e; font-weight:600;">
        <span>{mandante}</span><span>Empate</span><span>{visitante}</span>
    </div>
    """


def tag_liquidez(valor: float) -> str:
    if valor >= 2_000_000:
        return f'<span class="liquidity-tag liq-high">🔥 Alta Liquidez</span>'
    elif valor >= 500_000:
        return f'<span class="liquidity-tag liq-mid">💧 Liquidez Média</span>'
    return f'<span class="liquidity-tag liq-low">Liquidez Baixa</span>'


def grafico_probabilidades_1x2(probs: Dict[str, float], mandante: str, visitante: str):
    labels = [mandante, "Empate", visitante]
    valores = [probs["mandante"] * 100, probs["empate"] * 100, probs["visitante"] * 100]

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=valores,
        text=[f"{v:.1f}%" for v in valores],
        textposition="auto",
        marker_color=["#38ef7d", "#F7B731", "#FF5E62"],
        marker_line_color='rgba(0,0,0,0)',
        opacity=0.9
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=10, l=10, r=10),
        height=240,
        font=dict(color="#c9d1d9", family="Inter"),
        yaxis=dict(showgrid=True, gridcolor='#21262d', title=None, showticklabels=False),
        xaxis=dict(showgrid=False)
    )
    return fig


def render_detalhes_partida(partida: Partida, analise: dict):
    p = analise["probs_1x2"]
    ou = analise["over_under_gols"]
    esc = analise["escanteios"]
    sec = analise["secundarios"]
    top_10 = analise["top_10_placares"]

    st.markdown(tag_liquidez(partida.liquidez), unsafe_allow_html=True)
    st.markdown(render_prob_stacked_bar(p, partida.mandante, partida.visitante), unsafe_allow_html=True)
    st.write("")

    # Cabeçalho dos times com avatar + pills de estatística
    col_m, col_v = st.columns(2)
    with col_m:
        st.markdown(render_team_header(partida.mandante, "MANDANTE"), unsafe_allow_html=True)
        st.markdown(f"**Forma Recente:** {render_badge_forma(partida.estat_mandante.ultimos_5_jogos)}", unsafe_allow_html=True)
        st.markdown(render_stat_pills(partida.estat_mandante), unsafe_allow_html=True)

    with col_v:
        st.markdown(render_team_header(partida.visitante, "VISITANTE"), unsafe_allow_html=True)
        st.markdown(f"**Forma Recente:** {render_badge_forma(partida.estat_visitante.ultimos_5_jogos)}", unsafe_allow_html=True)
        st.markdown(render_stat_pills(partida.estat_visitante), unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Match Odds (1X2)", "📊 Top 10 Placares", "⚽ Gols", "🚩 Escanteios", "🟨 Cartões e Chutes"])

    with tab1:
        favorito = max(p, key=p.get)
        c1, c2, c3 = st.columns(3)
        c1.markdown(card_metrica_html(partida.mandante, f"{p['mandante']*100:.1f}%", f"Odd: {odd_justa(p['mandante'])}", destaque=(favorito == "mandante")), unsafe_allow_html=True)
        c2.markdown(card_metrica_html("Empate", f"{p['empate']*100:.1f}%", f"Odd: {odd_justa(p['empate'])}", destaque=(favorito == "empate")), unsafe_allow_html=True)
        c3.markdown(card_metrica_html(partida.visitante, f"{p['visitante']*100:.1f}%", f"Odd: {odd_justa(p['visitante'])}", destaque=(favorito == "visitante")), unsafe_allow_html=True)
        st.plotly_chart(grafico_probabilidades_1x2(p, partida.mandante, partida.visitante), use_container_width=True)

    with tab2:
        df_top10 = pd.DataFrame([
            {
                "Placar Exato": item["placar"],
                "Probabilidade": f"{item['prob_percentual']:.2f}%",
                "Odd Justa Estimada": f"{item['odd_justa']:.2f}"
            }
            for item in top_10
        ])
        st.dataframe(df_top10, use_container_width=True, hide_index=True)

    with tab3:
        cols = st.columns(len(ou))
        for col, (linha, valores) in zip(cols, ou.items()):
            with col:
                st.markdown(card_metrica_html(f"Over {linha}", f"{valores['over']*100:.1f}%", f"Odd: {odd_justa(valores['over'])}"), unsafe_allow_html=True)
                st.write("")
                st.markdown(card_metrica_html(f"Under {linha}", f"{valores['under']*100:.1f}%", f"Odd: {odd_justa(valores['under'])}"), unsafe_allow_html=True)

    with tab4:
        c1, c2, c3 = st.columns(3)
        c1.markdown(card_metrica_html(f"Cantos {partida.mandante}", str(esc["lambda_mandante"])), unsafe_allow_html=True)
        c2.markdown(card_metrica_html(f"Cantos {partida.visitante}", str(esc["lambda_visitante"])), unsafe_allow_html=True)
        c3.markdown(card_metrica_html("Total Esperado", str(esc["total_esperado"])), unsafe_allow_html=True)
        st.write("")
        st.markdown("**Over/Under Escanteios**")
        cols_ou_esc = st.columns(len(esc["over_under"]))
        for col, (linha, valores) in zip(cols_ou_esc, esc["over_under"].items()):
            with col:
                st.markdown(card_metrica_html(f"Over {linha}", f"{valores['over']*100:.1f}%", f"Odd: {odd_justa(valores['over'])}"), unsafe_allow_html=True)

    with tab5:
        c1, c2, c3 = st.columns(3)
        c1.markdown(card_metrica_html("Amarelos Esperados", str(sec["cartoes_amarelos"])), unsafe_allow_html=True)
        c2.markdown(card_metrica_html("Vermelhos Esperados", str(sec["cartoes_vermelhos"])), unsafe_allow_html=True)
        c3.markdown(card_metrica_html("Chutes no Gol", str(sec["finalizacoes"])), unsafe_allow_html=True)


def render_summary_strip(total_partidas: int, total_ligas: int, data_ref) -> str:
    return f"""
    <div class="summary-strip">
        <div class="summary-chip"><div class="num">{total_partidas}</div><div class="lbl">Partidas</div></div>
        <div class="summary-chip"><div class="num">{total_ligas}</div><div class="lbl">Competições</div></div>
        <div class="summary-chip"><div class="num">{data_ref.strftime('%d/%m')}</div><div class="lbl">Data Analisada</div></div>
    </div>
    """


def main():
    # Header com Visual Moderno
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">⚽ SILVER BOOL</div>
        <div class="hero-subtitle">Análise Preditiva & Inteligência Esportiva de Alto Desempenho</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Painel de Controle")
        modo_dados = st.radio("Fonte dos Dados", options=["Simulação", "API-Football"], index=0)

        api_key = ""
        if modo_dados == "API-Football":
            api_key = st.text_input("Cole sua API Key:", type="password", help="Insira sua chave da API-Football.")

        data_ref = st.date_input("Data dos Jogos", value=datetime.now(FUSO_BRASIL))
        st.divider()
        st.link_button("🌐 Consultar Flashscore", "https://www.flashscore.com.br/", use_container_width=True)
        st.markdown(
            '<div class="sidebar-footer">Modelo estatístico de Poisson.<br>Probabilidades geradas para fins analíticos.<br>Aposte com responsabilidade.</div>',
            unsafe_allow_html=True
        )

    data_ref_str = data_ref.strftime("%Y-%m-%d")

    with st.spinner("Carregando e processando estatísticas..."):
        partidas = obter_partidas_do_dia(data_ref_str, modo_dados, api_key)

        if not partidas:
            return

        ligas_dict: Dict[str, List[Partida]] = {}
        for p in partidas:
            ligas_dict.setdefault(p.liga_nome, []).append(p)

    st.markdown(render_summary_strip(len(partidas), len(ligas_dict), data_ref), unsafe_allow_html=True)

    for liga, lista_jogos in ligas_dict.items():
        st.markdown(f"""
        <div class="league-header">
            <span>🏆</span>
            <span class="league-name">{liga}</span>
            <span class="league-count">{len(lista_jogos)} jogo(s)</span>
        </div>
        """, unsafe_allow_html=True)
        lista_jogos_ordenada = sorted(lista_jogos, key=lambda p: p.horario)

        for p in lista_jogos_ordenada:
            analise = analisar_partida_completa(p)
            fogo = " 🔥" if p.liquidez >= 2_000_000 else ""
            titulo_expander = f"⏰ {p.horario.strftime('%H:%M')} — {p.mandante} x {p.visitante}{fogo}"

            with st.expander(titulo_expander):
                render_detalhes_partida(p, analise)


if __name__ == "__main__":
    main()
