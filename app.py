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
    page_title="SILVER BOOL",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    resultados = ["V", "E", "D"]
    pesos = [0.45, 0.30, 0.25]
    return rng.choices(resultados, weights=pesos, k=5)


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
    params = {"date": data}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=12)
        if resp.status_code == 200:
            res_json = resp.json()
            jogos = res_json.get("response", [])
            
            for item in jogos:
                pais = str(item['league']['country']).strip()
                liga_nome_bruto = str(item['league']['name']).strip()
                
                # Bloqueio do Campeonato e Times Russos
                if "russia" in pais.lower() or "russia" in liga_nome_bruto.lower() or "russian" in liga_nome_bruto.lower():
                    continue

                f_id = str(item["fixture"]["id"])
                liga_nome = f"{liga_nome_bruto} ({pais})"
                mandante_nome = item["teams"]["home"]["name"]
                visitante_nome = item["teams"]["away"]["name"]
                date_raw = item["fixture"]["date"]
                horario = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                
                seed_m = abs(hash(mandante_nome)) % 10000
                seed_v = abs(hash(visitante_nome)) % 10000
                
                estat_m = _gerar_estatisticas_aleatorias(mandante_nome, seed_m)
                estat_v = _gerar_estatisticas_aleatorias(visitante_nome, seed_v)
                
                rng_liq = random.Random(f_id)
                liquidez = round(rng_liq.uniform(50_000, 3_000_000), 2)
                
                partidas.append(Partida(
                    id=f_id,
                    liga_nome=liga_nome,
                    mandante=mandante_nome,
                    visitante=visitante_nome,
                    horario=horario,
                    liquidez=liquidez,
                    estat_mandante=estat_m,
                    estat_visitante=estat_v,
                ))
    except Exception as e:
        st.error(f"Erro na conexão com a API: {e}")
            
    return partidas


_TIMES_MUNDO = [
    ("Premier League (Inglaterra)", "Manchester City", "Arsenal"),
    ("Brasileirão Série A (Brasil)", "Palmeiras", "Flamengo"),
    ("La Liga (Espanha)", "Real Madrid", "Barcelona"),
    ("Serie A (Itália)", "Inter de Milão", "Juventus"),
    ("Bundesliga (Alemanha)", "Bayern de Munique", "Bayer Leverkusen"),
    ("Primera División (Argentina)", "Boca Juniors", "River Plate"),
    ("Primeira Liga (Portugal)", "Benfica", "Porto"),
    ("Süper Lig (Turquia)", "Galatasaray", "Fenerbahçe"),
]


@st.cache_data(ttl=3600, show_spinner=False)
def simular_partidas_do_dia(data_ref: str) -> List[Partida]:
    partidas = []
    seed_base = int(pd.Timestamp(data_ref).strftime("%Y%m%d"))
    
    for i, (liga, mandante, visitante) in enumerate(_TIMES_MUNDO):
        rng = random.Random(seed_base + i)
        estat_mandante = _gerar_estatisticas_aleatorias(mandante, seed_base + i)
        estat_visitante = _gerar_estatisticas_aleatorias(visitante, seed_base + i + 500)

        horario = datetime.strptime(data_ref, "%Y-%m-%d") + timedelta(
            hours=rng.choice([12, 14, 16, 17, 18, 20, 21]), minutes=rng.choice([0, 15, 30, 45])
        )
        liquidez = round(rng.uniform(150_000, 4_500_000), 2)

        partidas.append(Partida(
            id=f"sim_{i}",
            liga_nome=liga,
            mandante=mandante,
            visitante=visitante,
            horario=horario,
            liquidez=liquidez,
            estat_mandante=estat_mandante,
            estat_visitante=estat_visitante,
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


def formatar_forma(jogos: List[str]) -> str:
    icones = []
    for r in jogos:
        if r == "V": icones.append("🟩 V")
        elif r == "E": icones.append("🟨 E")
        else: icones.append("🟥 D")
    return " · ".join(icones)


def grafico_probabilidades_1x2(probs: Dict[str, float], mandante: str, visitante: str):
    labels = [mandante, "Empate", visitante]
    valores = [probs["mandante"] * 100, probs["empate"] * 100, probs["visitante"] * 100]
    fig = go.Figure(data=[go.Bar(x=labels, y=valores, text=[f"{v:.1f}%" for v in valores], textposition="auto", marker_color=["#2E86DE", "#F1C40F", "#E74C3C"])])
    fig.update_layout(title="Probabilidades Match Odds (1X2)", yaxis_title="Probabilidade (%)", height=320, margin=dict(t=40, b=20))
    return fig


def grafico_over_under(dados_ou: Dict[float, Dict[str, float]], titulo: str):
    linhas = list(dados_ou.keys())
    fig = go.Figure(data=[
        go.Bar(name="Over", x=[str(l) for l in linhas], y=[dados_ou[l]["over"] * 100 for l in linhas], marker_color="#27AE60"),
        go.Bar(name="Under", x=[str(l) for l in linhas], y=[dados_ou[l]["under"] * 100 for l in linhas], marker_color="#C0392B"),
    ])
    fig.update_layout(barmode="group", title=titulo, yaxis_title="Probabilidade (%)", height=320)
    return fig


def render_detalhes_partida(partida: Partida, analise: dict):
    p = analise["probs_1x2"]
    ou = analise["over_under_gols"]
    esc = analise["escanteios"]
    sec = analise["secundarios"]

    # --- Seção 1: Forma Recente e Médias ---
    st.markdown("#### 📊 Retrospecto dos ÚLTIMOS 5 JOGOS e Estatísticas Média")
    col_m, col_v = st.columns(2)
    
    with col_m:
        st.markdown(f"**🏠 {partida.mandante}**")
        st.markdown(f"**Forma Recente:** {formatar_forma(partida.estat_mandante.ultimos_5_jogos)}")
        st.caption(f"⚽ Gols Marcados (média): {partida.estat_mandante.gols_marcados_media} | Sofridos: {partida.estat_mandante.gols_sofridos_media}")
        st.caption(f"🚩 Cantos Pró (média): {partida.estat_mandante.escanteios_favor_media} | Cartões Amarelos: {partida.estat_mandante.cartoes_amarelos_media}")

    with col_v:
        st.markdown(f"**🚀 {partida.visitante}**")
        st.markdown(f"**Forma Recente:** {formatar_forma(partida.estat_visitante.ultimos_5_jogos)}")
        st.caption(f"⚽ Gols Marcados (média): {partida.estat_visitante.gols_marcados_media} | Sofridos: {partida.estat_visitante.gols_sofridos_media}")
        st.caption(f"🚩 Cantos Pró (média): {partida.estat_visitante.escanteios_favor_media} | Cartões Amarelos: {partida.estat_visitante.cartoes_amarelos_media}")

    st.divider()

    # --- Seção 2: Probabilidades e Odds Justas ---
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Match Odds (1X2)", "⚽ Gols", "🚩 Escanteios", "🟨 Cartões e Chutes"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Vitória {partida.mandante}", f"{p['mandante']*100:.1f}%", f"Odd justa: {odd_justa(p['mandante'])}")
        c2.metric("Empate", f"{p['empate']*100:.1f}%", f"Odd justa: {odd_justa(p['empate'])}")
        c3.metric(f"Vitória {partida.visitante}", f"{p['visitante']*100:.1f}%", f"Odd justa: {odd_justa(p['visitante'])}")
        st.plotly_chart(grafico_probabilidades_1x2(p, partida.mandante, partida.visitante), use_container_width=True)

    with tab2:
        cols = st.columns(len(ou))
        for col, (linha, valores) in zip(cols, ou.items()):
            with col:
                st.metric(f"Over {linha}", f"{valores['over']*100:.1f}%", f"Odd: {odd_justa(valores['over'])}")
                st.metric(f"Under {linha}", f"{valores['under']*100:.1f}%", f"Odd: {odd_justa(valores['under'])}")
        st.plotly_chart(grafico_over_under(ou, "Probabilidade de Gols"), use_container_width=True)

    with tab3:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Cantos {partida.mandante}", esc["lambda_mandante"])
        c2.metric(f"Cantos {partida.visitante}", esc["lambda_visitante"])
        c3.metric("Total Esperado", esc["total_esperado"])
        st.plotly_chart(grafico_over_under(esc["over_under"], "Probabilidade de Escanteios"), use_container_width=True)

    with tab4:
        c1, c2, c3 = st.columns(3)
        c1.metric("Amarelos Esperados", sec["cartoes_amarelos_esperados"])
        c2.metric("Vermelhos Esperados", sec["cartoes_vermelhos_esperados"])
        c3.metric("Finalizações no Gol", sec["finalizacoes_gol_esperadas"])


def main():
    st.title("⚽ SILVER BOOL")
    st.caption("Análise Preditiva e Estatísticas de Apostas Esportivas · Cobertura Total de Jogos")

    with st.sidebar:
        st.header("⚙️ SILVER BOOL — Painel")
        modo_dados = st.radio("Fonte de dados", options=["Simulação", "API-Football"], index=0)
        
        api_key = ""
        if modo_dados == "API-Football":
            api_key = st.text_input("Cole sua API Key aqui:", type="password", help="Insira sua chave da API-Football.")

        data_ref = st.date_input("Data de referência", value=datetime.now())
        st.divider()
        
        st.markdown("**🔗 Atalhos Externos:**")
        st.link_button("🌐 Consultar Flashscore", "https://www.flashscore.com.br/", use_container_width=True)

    data_ref_str = data_ref.strftime("%Y-%m-%d")

    with st.spinner("SILVER BOOL: Carregando todos os jogos da data..."):
        partidas = obter_partidas_do_dia(data_ref_str, modo_dados, api_key)
        
        if not partidas:
            return

        partidas_ordenadas = sorted(partidas, key=lambda p: p.horario)

    st.subheader(f"📋 Lista Completa de Jogos do Dia ({len(partidas_ordenadas)} partidas encontradas)")
    st.info("💡 Clique sobre qualquer partida abaixo para expandir e visualizar as estatísticas completas, os últimos 5 jogos e as probabilidades.")

    for p in partidas_ordenadas:
        analise = analisar_partida_completa(p)
        titulo_expander = f"⏰ {p.horario.strftime('%H:%M')} — [{p.liga_nome}] {p.mandante} x {p.visitante}"
        
        with st.expander(titulo_expander):
            render_detalhes_partida(p, analise)


if __name__ == "__main__":
    main()
