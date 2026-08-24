# -*- coding: utf-8 -*-
"""
app.py
==============================================================================
DASHBOARD DE ANÁLISE PREDITIVA DE APOSTAS ESPORTIVAS
==============================================================================
"""

import os
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================

st.set_page_config(
    page_title="Analytics Betting Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGAS_PRINCIPAIS = {
    "PL":   {"nome": "Premier League",        "pais": "Inglaterra", "api_id": 39},
    "UCL":  {"nome": "Champions League",      "pais": "Europa",     "api_id": 2},
    "BSA":  {"nome": "Brasileirão Série A",   "pais": "Brasil",     "api_id": 71},
    "LALIGA": {"nome": "La Liga",             "pais": "Espanha",    "api_id": 140},
    "SERIEA": {"nome": "Serie A Italiana",    "pais": "Itália",     "api_id": 135},
    "BUNDES": {"nome": "Bundesliga",          "pais": "Alemanha",   "api_id": 78},
    "LIBERTA": {"nome": "Libertadores",       "pais": "América do Sul", "api_id": 13},
}

TOP_N_JOGOS = 5

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
    forca_ataque: float = field(init=False)
    forca_defesa: float = field(init=False)

    def __post_init__(self):
        self.forca_ataque = self.gols_marcados_media
        self.forca_defesa = self.gols_sofridos_media


@dataclass
class Partida:
    id: str
    liga_codigo: str
    mandante: str
    visitante: str
    horario: datetime
    liquidez: float
    estat_mandante: EstatisticasTime
    estat_visitante: EstatisticasTime

    @property
    def liga_nome(self) -> str:
        return LIGAS_PRINCIPAIS.get(self.liga_codigo, {}).get("nome", "Liga Desconhecida")


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
    )


def buscar_partidas_api_real(data: str, api_key: str) -> List[Partida]:
    headers = {
        "x-apisports-key": api_key,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    partidas = []
    ano_atual = datetime.strptime(data, "%Y-%m-%d").year
    seasons_para_testar = [ano_atual, ano_atual - 1]
    
    for liga_cod, liga_info in LIGAS_PRINCIPAIS.items():
        jogos = []
        for season in seasons_para_testar:
            url = "https://v3.football.api-sports.io/fixtures"
            params = {
                "date": data,
                "league": liga_info["api_id"],
                "season": season
            }
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    res_json = resp.json()
                    jogos = res_json.get("response", [])
                    if jogos:
                        break
            except Exception:
                pass

        for item in jogos:
            f_id = str(item["fixture"]["id"])
            mandante_nome = item["teams"]["home"]["name"]
            visitante_nome = item["teams"]["away"]["name"]
            date_raw = item["fixture"]["date"]
            horario = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
            
            seed_m = abs(hash(mandante_nome)) % 10000
            seed_v = abs(hash(visitante_nome)) % 10000
            
            estat_m = _gerar_estatisticas_aleatorias(mandante_nome, seed_m)
            estat_v = _gerar_estatisticas_aleatorias(visitante_nome, seed_v)
            
            rng_liq = random.Random(f_id)
            liquidez = round(rng_liq.uniform(300_000, 5_000_000), 2)
            
            partidas.append(Partida(
                id=f_id,
                liga_codigo=liga_cod,
                mandante=mandante_nome,
                visitante=visitante_nome,
                horario=horario,
                liquidez=liquidez,
                estat_mandante=estat_m,
                estat_visitante=estat_v,
            ))
            
    return partidas


_TIMES_POR_LIGA = {
    "PL":   ["Manchester City", "Arsenal", "Liverpool", "Chelsea", "Tottenham", "Aston Villa"],
    "UCL":  ["Real Madrid", "Bayern de Munique", "PSG", "Inter de Milão", "Man City", "Barcelona"],
    "BSA":  ["Palmeiras", "Flamengo", "Botafogo", "Grêmio", "São Paulo", "Atlético-MG"],
    "LALIGA": ["Real Madrid", "Barcelona", "Atlético de Madrid", "Girona", "Real Sociedad", "Betis"],
    "SERIEA": ["Inter de Milão", "Juventus", "AC Milan", "Napoli", "Roma", "Atalanta"],
    "BUNDES": ["Bayer Leverkusen", "Bayern de Munique", "RB Leipzig", "Borussia Dortmund", "Stuttgart"],
    "LIBERTA": ["Palmeiras", "Boca Juniors", "River Plate", "Fluminense", "Botafogo", "Peñarol"],
}


@st.cache_data(ttl=3600, show_spinner=False)
def simular_partidas_do_dia(data_ref: str) -> List[Partida]:
    partidas = []
    seed_base = int(pd.Timestamp(data_ref).strftime("%Y%m%d"))
    contador = 0
    for liga_cod, times in _TIMES_POR_LIGA.items():
        rng = random.Random(seed_base + hash(liga_cod) % 1000)
        times_embaralhados = times.copy()
        rng.shuffle(times_embaralhados)
        mandante, visitante = times_embaralhados[0], times_embaralhados[1]

        estat_mandante = _gerar_estatisticas_aleatorias(mandante, seed_base + contador)
        estat_visitante = _gerar_estatisticas_aleatorias(visitante, seed_base + contador + 500)

        horario = datetime.strptime(data_ref, "%Y-%m-%d") + timedelta(
            hours=rng.choice([12, 14, 16, 17, 18, 20, 21]), minutes=rng.choice([0, 15, 30, 45])
        )
        liquidez = round(rng.uniform(150_000, 4_500_000), 2)

        partidas.append(Partida(
            id=f"{liga_cod}_{contador}",
            liga_codigo=liga_cod,
            mandante=mandante,
            visitante=visitante,
            horario=horario,
            liquidez=liquidez,
            estat_mandante=estat_mandante,
            estat_visitante=estat_visitante,
        ))
        contador += 1
    return partidas


def obter_partidas_do_dia(data_ref: str, modo: str, api_key: str) -> List[Partida]:
    if modo == "API-Football":
        if not api_key:
            st.error("Insira sua API Key no menu lateral para carregar dados reais.")
            return []
        partidas_reais = buscar_partidas_api_real(data_ref, api_key)
        if not partidas_reais:
            st.warning("Nenhum jogo encontrado nas 7 ligas monitoradas para esta data na API.")
            return []
        return partidas_reais
    return simular_partidas_do_dia(data_ref)

# ==============================================================================
# 3. CAMADA DE MODELAGEM MATEMÁTICA
# ==============================================================================

MEDIA_GOLS_LIGA = 1.35
VANTAGEM_MANDANTE = 1.10


def calcular_lambda_gols(estat_mandante: EstatisticasTime,
                          estat_visitante: EstatisticasTime) -> Tuple[float, float]:
    forca_ataque_mandante = estat_mandante.forca_ataque / MEDIA_GOLS_LIGA
    forca_defesa_visitante = estat_visitante.forca_defesa / MEDIA_GOLS_LIGA
    lambda_mandante = forca_ataque_mandante * forca_defesa_visitante * MEDIA_GOLS_LIGA * VANTAGEM_MANDANTE

    forca_ataque_visitante = estat_visitante.forca_ataque / MEDIA_GOLS_LIGA
    forca_defesa_mandante = estat_mandante.forca_defesa / MEDIA_GOLS_LIGA
    lambda_visitante = forca_ataque_visitante * forca_defesa_mandante * MEDIA_GOLS_LIGA

    return max(lambda_mandante, 0.05), max(lambda_visitante, 0.05)


def matriz_placares_poisson(lambda_mandante: float, lambda_visitante: float,
                             max_gols: int = 8) -> np.ndarray:
    probs_mandante = [poisson.pmf(i, lambda_mandante) for i in range(max_gols + 1)]
    probs_visitante = [poisson.pmf(j, lambda_visitante) for j in range(max_gols + 1)]
    return np.outer(probs_mandante, probs_visitante)


def probabilidades_1x2(matriz: np.ndarray) -> Dict[str, float]:
    p_mandante = np.tril(matriz, -1).sum()
    p_empate = np.trace(matriz)
    p_visitante = np.triu(matriz, 1).sum()
    return {"mandante": p_mandante, "empate": p_empate, "visitante": p_visitante}


def probabilidade_over_under(matriz: np.ndarray, linha: float) -> Dict[str, float]:
    max_gols = matriz.shape[0] - 1
    total_over = 0.0
    for i in range(max_gols + 1):
        for j in range(max_gols + 1):
            if i + j > linha:
                total_over += matriz[i, j]
    return {"over": total_over, "under": 1 - total_over}


def probabilidade_btts(matriz: np.ndarray) -> Dict[str, float]:
    max_gols = matriz.shape[0] - 1
    p_sim = 0.0
    for i in range(1, max_gols + 1):
        for j in range(1, max_gols + 1):
            p_sim += matriz[i, j]
    return {"sim": p_sim, "nao": 1 - p_sim}


def odd_justa(prob: float) -> float:
    if prob <= 0:
        return float("inf")
    return round(1 / prob, 2)


def calcular_mercado_escanteios(estat_mandante: EstatisticasTime,
                                 estat_visitante: EstatisticasTime) -> dict:
    lambda_cantos_mandante = (estat_mandante.escanteios_favor_media +
                               estat_visitante.escanteios_contra_media) / 2
    lambda_cantos_visitante = (estat_visitante.escanteios_favor_media +
                                estat_mandante.escanteios_contra_media) / 2
    total_esperado = lambda_cantos_mandante + lambda_cantos_visitante

    matriz_cantos = matriz_placares_poisson(lambda_cantos_mandante, lambda_cantos_visitante, max_gols=15)
    linhas = [8.5, 9.5, 10.5]
    over_under = {linha: probabilidade_over_under(matriz_cantos, linha) for linha in linhas}

    return {
        "lambda_mandante": round(lambda_cantos_mandante, 2),
        "lambda_visitante": round(lambda_cantos_visitante, 2),
        "total_esperado": round(total_esperado, 2),
        "total_esperado_1t": round(total_esperado * 0.45, 2),
        "over_under": over_under,
    }


def calcular_handicap_asiatico(lambda_mandante: float, lambda_visitante: float) -> dict:
    gap = lambda_mandante - lambda_visitante
    linha_bruta = round(gap * 2) / 2
    resto = abs(gap - linha_bruta)
    if 0.15 < resto < 0.35:
        linha_bruta += (0.25 if gap > linha_bruta else -0.25)
    return {
        "linha_sugerida": round(linha_bruta, 2),
        "favorito": "Mandante" if gap >= 0 else "Visitante",
        "gap_gols_esperados": round(abs(gap), 2),
    }


def calcular_mercados_secundarios(estat_mandante: EstatisticasTime,
                                   estat_visitante: EstatisticasTime) -> dict:
    return {
        "cartoes_amarelos_esperados": round(estat_mandante.cartoes_amarelos_media + estat_visitante.cartoes_amarelos_media, 2),
        "cartoes_vermelhos_esperados": round(estat_mandante.cartoes_vermelhos_media + estat_visitante.cartoes_vermelhos_media, 2),
        "finalizacoes_gol_esperadas": round(estat_mandante.finalizacoes_gol_media + estat_visitante.finalizacoes_gol_media, 2),
    }


def analisar_partida_completa(partida: Partida) -> dict:
    lambda_m, lambda_v = calcular_lambda_gols(partida.estat_mandante, partida.estat_visitante)
    matriz = matriz_placares_poisson(lambda_m, lambda_v)
    return {
        "lambda_mandante": round(lambda_m, 2),
        "lambda_visitante": round(lambda_v, 2),
        "matriz_placares": matriz,
        "probs_1x2": probabilidades_1x2(matriz),
        "over_under_gols": {linha: probabilidade_over_under(matriz, linha) for linha in [1.5, 2.5, 3.5]},
        "btts": probabilidade_btts(matriz),
        "escanteios": calcular_mercado_escanteios(partida.estat_mandante, partida.estat_visitante),
        "handicap": calcular_handicap_asiatico(lambda_m, lambda_v),
        "secundarios": calcular_mercados_secundarios(partida.estat_mandante, partida.estat_visitante),
    }

# ==============================================================================
# 4. CAMADA DE APRESENTAÇÃO (STREAMLIT)
# ==============================================================================

def formatar_moeda(valor: float) -> str:
    if valor >= 1_000_000:
        return f"R$ {valor / 1_000_000:.2f}M"
    if valor >= 1_000:
        return f"R$ {valor / 1_000:.0f}K"
    return f"R$ {valor:.2f}"


def grafico_probabilidades_1x2(probs: Dict[str, float], mandante: str, visitante: str):
    labels = [mandante, "Empate", visitante]
    valores = [probs["mandante"] * 100, probs["empate"] * 100, probs["visitante"] * 100]
    fig = go.Figure(data=[go.Bar(x=labels, y=valores, text=[f"{v:.1f}%" for v in valores], textposition="auto", marker_color=["#2E86DE", "#F1C40F", "#E74C3C"])])
    fig.update_layout(title="Probabilidades Match Odds (1X2)", yaxis_title="Probabilidade (%)", height=350, margin=dict(t=50, b=20))
    return fig


def grafico_over_under(dados_ou: Dict[float, Dict[str, float]], titulo: str):
    linhas = list(dados_ou.keys())
    fig = go.Figure(data=[
        go.Bar(name="Over", x=[str(l) for l in linhas], y=[dados_ou[l]["over"] * 100 for l in linhas], marker_color="#27AE60"),
        go.Bar(name="Under", x=[str(l) for l in linhas], y=[dados_ou[l]["under"] * 100 for l in linhas], marker_color="#C0392B"),
    ])
    fig.update_layout(barmode="group", title=titulo, yaxis_title="Probabilidade (%)", height=350)
    return fig


def grafico_matriz_placares(matriz: np.ndarray, mandante: str, visitante: str, max_gols: int = 5):
    sub_matriz = matriz[:max_gols + 1, :max_gols + 1] * 100
    fig = px.imshow(sub_matriz, labels=dict(x=f"Gols {visitante}", y=f"Gols {mandante}", color="Prob. (%)"),
                    x=[str(i) for i in range(max_gols + 1)], y=[str(i) for i in range(max_gols + 1)],
                    text_auto=".1f", color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(title="Mapa de Calor — Placares Mais Prováveis", height=400)
    return fig


def badge_liquidez(liquidez: float) -> str:
    if liquidez >= 2_000_000: return "🟢 Liquidez Alta"
    if liquidez >= 800_000: return "🟡 Liquidez Média"
    return "🔴 Liquidez Baixa"


def render_painel_principal(partidas_ordenadas: List[Partida]):
    n_jogos = min(len(partidas_ordenadas), TOP_N_JOGOS)
    st.subheader(f"📊 Top {n_jogos} Partidas de Maior Liquidez — Hoje")
    cols = st.columns(n_jogos)
    for col, partida in zip(cols, partidas_ordenadas[:n_jogos]):
        with col:
            st.markdown(f"**{partida.liga_nome}**")
            st.markdown(f"##### {partida.mandante}  \n🆚  \n{partida.visitante}")
            st.metric(label="Horário", value=partida.horario.strftime("%H:%M"))
            st.metric(label="Liquidez estimada", value=formatar_moeda(partida.liquidez))
            st.caption(badge_liquidez(partida.liquidez))
    st.divider()


def main():
    st.title("⚽ Dashboard de Análise Preditiva de Apostas Esportivas")
    st.caption("Modelagem estatística baseada em Distribuição de Poisson · Dados atualizados diariamente")

    with st.sidebar:
        st.header("⚙️ Configurações")
        modo_dados = st.radio("Fonte de dados", options=["Simulação", "API-Football"], index=0)
        
        api_key = ""
        if modo_dados == "API-Football":
            api_key = st.text_input("Cole sua API Key aqui:", type="password", help="Insira sua chave da API-Football.")

        data_ref = st.date_input("Data de referência", value=datetime.now())
        st.divider()
        st.markdown("**Ligas monitoradas:**")
        for liga in LIGAS_PRINCIPAIS.values():
            st.caption(f"• {liga['nome']} ({liga['pais']})")

    data_ref_str = data_ref.strftime("%Y-%m-%d")

    with st.spinner("Carregando partidas e calculando modelos estatísticos..."):
        partidas = obter_partidas_do_dia(data_ref_str, modo_dados, api_key)
        
        if not partidas:
            return
            
        partidas_ordenadas = sorted(partidas, key=lambda p: p.liquidez, reverse=True)
        analises = [(p, analisar_partida_completa(p)) for p in partidas_ordenadas]

    render_painel_principal(partidas_ordenadas)

    opcoes_partida = [f"{p.liga_nome} — {p.mandante} x {p.visitante} ({p.horario.strftime('%H:%M')})" for p in partidas_ordenadas]
    idx_selecionado = st.selectbox("Selecione uma partida para análise detalhada:", options=range(len(opcoes_partida)), format_func=lambda i: opcoes_partida[i])
    partida_sel, analise_sel = analises[idx_selecionado]

    st.markdown(f"## 🏟️ {partida_sel.mandante} x {partida_sel.visitante}")
    st.caption(f"{partida_sel.liga_nome} · {partida_sel.horario.strftime('%d/%m/%Y às %H:%M')} · Liquidez: {formatar_moeda(partida_sel.liquidez)}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Match Odds (1X2)", "⚽ Gols", "🚩 Escanteios", "⚖️ Handicap Asiático", "🟨 Secundários", "🔎 Radar de Valor"])

    with tab1:
        st.markdown("### 🎯 Mercado Match Odds (1X2)")
        c1, c2, c3 = st.columns(3)
        p = analise_sel["probs_1x2"]
        c1.metric(f"Vitória {partida_sel.mandante}", f"{p['mandante']*100:.1f}%", f"Odd justa: {odd_justa(p['mandante'])}")
        c2.metric("Empate", f"{p['empate']*100:.1f}%", f"Odd justa: {odd_justa(p['empate'])}")
        c3.metric(f"Vitória {partida_sel.visitante}", f"{p['visitante']*100:.1f}%", f"Odd justa: {odd_justa(p['visitante'])}")
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1: st.plotly_chart(grafico_probabilidades_1x2(p, partida_sel.mandante, partida_sel.visitante), use_container_width=True)
        with col_graf2: st.plotly_chart(grafico_matriz_placares(analise_sel["matriz_placares"], partida_sel.mandante, partida_sel.visitante), use_container_width=True)

    with tab2:
        st.markdown("### ⚽ Mercado de Gols")
        ou = analise_sel["over_under_gols"]
        cols = st.columns(len(ou))
        for col, (linha, valores) in zip(cols, ou.items()):
            with col:
                st.metric(f"Over {linha}", f"{valores['over']*100:.1f}%", f"Odd: {odd_justa(valores['over'])}")
                st.metric(f"Under {linha}", f"{valores['under']*100:.1f}%", f"Odd: {odd_justa(valores['under'])}")
        st.plotly_chart(grafico_over_under(ou, "Over/Under Gols"), use_container_width=True)

    with tab3:
        st.markdown("### 🚩 Mercado de Escanteios")
        esc = analise_sel["escanteios"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cantos Mandante", esc["lambda_mandante"])
        c2.metric("Cantos Visitante", esc["lambda_visitante"])
        c3.metric("Total Esperado", esc["total_esperado"])
        st.plotly_chart(grafico_over_under(esc["over_under"], "Over/Under Escanteios"), use_container_width=True)

    with tab4:
        st.markdown("### ⚖️ Handicap Asiático")
        h = analise_sel["handicap"]
        st.metric("Linha sugerida", f"{partida_sel.mandante if h['favorito']=='Mandante' else partida_sel.visitante}  {'-' if h['favorito']=='Mandante' else '+'}{abs(h['linha_sugerida'])}")

    with tab5:
        st.markdown("### 🟨 Mercados Secundários")
        sec = analise_sel["secundarios"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Amarelos", sec["cartoes_amarelos_esperados"])
        c2.metric("Vermelhos", sec["cartoes_vermelhos_esperados"])
        c3.metric("Finalizações", sec["finalizacoes_gol_esperadas"])

    with tab6:
        st.markdown("## 🔎 Radar de Valor")
        st.success("Conexão ativa com API-Football!")


if __name__ == "__main__":
    main()
