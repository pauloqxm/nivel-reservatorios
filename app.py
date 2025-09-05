# app.py
import re
import math
import unicodedata
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Reservatórios – Tabela diária", layout="wide")

# ==========================
# Configuração
# ==========================
SHEETS_URL = "https://docs.google.com/spreadsheets/d/1zZ0RCyYj-AzA_dhWzxRziDWjgforbaH7WIoSEd2EKdk/edit?gid=1305065127#gid=1305065127"

# ==========================
# Utilitários
# ==========================
@st.cache_data(ttl=900)
def google_sheets_to_csv_url(url: str) -> str:
    """Converte URL do Google Sheets em URL CSV export."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url or "")
    gid_match = re.search(r"[#?]gid=(\d+)", url or "")
    gid = gid_match.group(1) if gid_match else None
    if not m:
        return url
    doc_id = m.group(1)
    base = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
    return f"{base}&gid={gid}" if gid else base

def strip_accents_lower(s: str) -> str:
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.lower()

def to_number(x):
    """'1.234,56' -> 1234.56; preserva NaN."""
    if x is None:
        return math.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return math.nan
    s = ''.join(ch for ch in s if ch.isdigit() or ch in ".,-")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return math.nan

def to_datetime_any(x):
    """Converte datas em vários formatos, retornando Timestamp ou NaT."""
    if pd.isna(x):
        return pd.NaT
    try:
        # Tenta converter removendo timezone se presente
        if isinstance(x, str) and 'T' in x:
            x = x.split('T')[0]
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

@st.cache_data(ttl=900)
def load_data_from_url(url: str) -> pd.DataFrame:
    csv_url = google_sheets_to_csv_url(url)
    df = pd.read_csv(csv_url, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_column(df: pd.DataFrame, aliases):
    """Retorna o nome da coluna cujo normalizado bate com um dos aliases."""
    aliases = set(list(aliases))
    normalized = {col: strip_accents_lower(col) for col in df.columns}
    # Exato
    for col, norm in normalized.items():
        if norm in aliases:
            return col
    # Contém
    for col, norm in normalized.items():
        if any(alias in norm for alias in aliases):
            return col
    return None

# ==========================
# Cálculo principal - CORREÇÃO PRINCIPAL
# ==========================
def compute_table_global_dates(df_raw: pd.DataFrame) -> pd.DataFrame:
    def last_scalar_on_date(dfr: pd.DataFrame, date_col: str, target_date, value_col: str) -> float:
        """
        Retorna o último valor numérico (float) para a data exata target_date.
        Garante SEMPRE um escalar (ou NaN).
        """
        if pd.isna(target_date):
            return math.nan
        
        # Converte ambas as datas para o mesmo formato (apenas data, sem hora)
        try:
            # Garante que target_date seja apenas data
            target_date_only = pd.Timestamp(target_date).normalize()
            
            # Converte a coluna de datas para o mesmo formato
            dfr_dates = pd.to_datetime(dfr[date_col]).dt.normalize()
            
            # Encontra registros com a mesma data
            mask = dfr_dates == target_date_only
            sel = dfr.loc[mask, value_col]
            
            if sel.empty:
                return math.nan
                
            # Converte para numérico e pega o último valor válido
            sel_numeric = pd.to_numeric(sel, errors='coerce').dropna()
            if sel_numeric.empty:
                return math.nan
                
            return float(sel_numeric.iloc[-1])
            
        except Exception:
            return math.nan

    df = df_raw.copy()

    # Mapear colunas por nomes comuns/aliases
    col_reservatorio = find_column(df, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    col_cota_sangria = find_column(df, {"cota sangria", "cota de sangria", "cota_sangria", "cota excedencia"})
    col_data         = find_column(df, {"data", "dt", "dia"})
    col_volume       = find_column(df, {"volume", "vol"})
    col_percentual   = find_column(df, {"percentual", "perc", "percentual (%)", "volume (%)"})
    col_nivel        = find_column(df, {"nivel", "nível", "cota", "altura"})

    required = {
        "Reservatório": col_reservatorio,
        "Cota Sangria": col_cota_sangria,
        "Data": col_data,
        "Volume": col_volume,
        "Percentual": col_percentual,
        "Nivel": col_nivel,
    }
    missing = [k for k, v in required.items() if v is None]
    if len(missing) > 0:
        raise ValueError(
            "Não foi possível identificar as colunas na planilha. "
            f"Faltando: {', '.join(missing)}. "
            "Renomeie na planilha ou ajuste os aliases no código."
        )

    # Conversões robustas
    df[col_data]         = df[col_data].apply(to_datetime_any)
    df[col_volume]       = df[col_volume].apply(to_number)
    df[col_percentual]   = df[col_percentual].apply(to_number)
    df[col_nivel]        = df[col_nivel].apply(to_number)
    df[col_cota_sangria] = df[col_cota_sangria].apply(to_number)

    df = df.dropna(subset=[col_data])
    if len(df) == 0:
        return pd.DataFrame()

    # Datas globais mais recentes (apenas a parte da data, sem hora)
    unique_dates = pd.Series(df[col_data].dropna().unique())
    unique_dates = pd.to_datetime(unique_dates, errors="coerce").dropna()
    unique_dates = unique_dates.dt.normalize().unique()  # Remove duplicatas e normaliza
    unique_dates = sorted(unique_dates)
    
    if len(unique_dates) == 0:
        return pd.DataFrame()
        
    data_atual    = unique_dates[-1]
    data_anterior = unique_dates[-2] if len(unique_dates) >= 2 else pd.NaT

    # Rótulos "dd/mm/aaaa"
    col_atual_label    = data_atual.strftime("%d/%m/%Y") if pd.notna(data_atual) else "Data Atual"
    col_anterior_label = data_anterior.strftime("%d/%m/%Y") if pd.notna(data_anterior) else "Data Anterior"

    rows = []
    for res, dfr in df.groupby(col_reservatorio, dropna=True):
        # NÍVEIS nas datas globais
        nivel_atual    = last_scalar_on_date(dfr, col_data, data_atual, col_nivel)
        nivel_anterior = last_scalar_on_date(dfr, col_data, data_anterior, col_nivel) if pd.notna(data_anterior) else math.nan

        # Volume/Percentual do dia ATUAL para capacidade total
        vol_atual  = last_scalar_on_date(dfr, col_data, data_atual, col_volume)
        perc_atual = last_scalar_on_date(dfr, col_data, data_atual, col_percentual)
        if pd.notna(perc_atual) and perc_atual != 0 and pd.notna(vol_atual):
            cap_total = vol_atual / (perc_atual / 100.0)
        else:
            cap_total = math.nan

        # Cota de sangria (preferir no dia atual; senão último histórico não-nulo)
        cota_atual = last_scalar_on_date(dfr, col_data, data_atual, col_cota_sangria)
        if pd.notna(cota_atual):
            cota_sangria_val = cota_atual
        else:
            cota_hist = pd.to_numeric(dfr[col_cota_sangria], errors="coerce").dropna()
            cota_sangria_val = float(cota_hist.iloc[-1]) if len(cota_hist) > 0 else math.nan

        variacao = (nivel_atual - nivel_anterior) if (pd.notna(nivel_atual) and pd.notna(nivel_anterior)) else math.nan

        rows.append({
            "Reservatório": res,
            "Cota Sangria": cota_sangria_val,
            col_anterior_label: nivel_anterior,
            col_atual_label: nivel_atual,
            "Capacidade Total (m³)": cap_total,
            "Variação do Nível": variacao,
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        order = ["Reservatório", "Cota Sangria", col_anterior_label, col_atual_label, "Capacidade Total (m³)", "Variação do Nível"]
        # Mantém apenas colunas que existem no DataFrame
        order = [col for col in order if col in out.columns]
        out = out.reindex(columns=order).sort_values("Reservatório").reset_index(drop=True)
    return out

# ==========================
# UI
# ==========================
st.title("📊 Tabela diária de Reservatórios")

with st.sidebar:
    st.markdown("### Fonte dos dados")
    default_mode = st.radio(
        "Escolha a fonte",
        ("Google Sheets (link padrão)", "Enviar CSV (arquivo local)"),
        index=0,
    )
    if default_mode == "Google Sheets (link padrão)":
        url = st.text_input("URL do Google Sheets", value=SHEETS_URL)
        uploaded_file = None
    else:
        url = None
        uploaded_file = st.file_uploader("Envie um CSV com a mesma estrutura", type=["csv"])

try:
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, dtype=str)
        st.success(f"CSV carregado: {uploaded_file.name}")
    else:
        df_raw = load_data_from_url(url)
        st.success("Dados carregados do Google Sheets")

    # Mostra prévia dos dados
    with st.expander("Visualizar dados brutos"):
        st.dataframe(df_raw.head(), use_container_width=True)

    # Filtro opcional por reservatório
    col_res_guess = find_column(df_raw, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    if col_res_guess is not None:
        reservatorios = sorted([x for x in df_raw[col_res_guess].dropna().unique() if x])
        sel = st.multiselect("Filtrar reservatórios (opcional)", reservatorios, [])
        df_filtered = df_raw[df_raw[col_res_guess].isin(sel)] if len(sel) > 0 else df_raw
    else:
        df_filtered = df_raw
        st.warning("Não foi possível identificar a coluna de reservatórios")

    # Calcula tabela final
    with st.spinner("Processando dados..."):
        result = compute_table_global_dates(df_filtered)

    st.subheader("Resultado")
    if result is None or len(result) == 0:
        st.info("Nenhum dado com as duas datas mais recentes foi encontrado.")
    else:
        # Formatação amigável pt-BR
        result_fmt = result.copy()
        for col in result_fmt.columns:
            if col == "Reservatório":
                continue
            if col == "Capacidade Total (m³)":
                result_fmt[col] = result_fmt[col].apply(
                    lambda v: f"{v:,.0f}".replace(",", "temp").replace(".", ",").replace("temp", ".") 
                    if pd.notna(v) else ""
                )
            elif col in ("Cota Sangria", "Variação do Nível"):
                result_fmt[col] = result_fmt[col].apply(
                    lambda v: f"{v:,.2f}".replace(",", "temp").replace(".", ",").replace("temp", ".") 
                    if pd.notna(v) else ""
                )
            else:
                # Colunas de data (valores de Nível)
                result_fmt[col] = result_fmt[col].apply(
                    lambda v: f"{v:,.2f}".replace(",", "temp").replace(".", ",").replace("temp", ".") 
                    if pd.notna(v) else ""
                )

        st.dataframe(result_fmt, use_container_width=True, hide_index=True)

        # Download CSV bruto
        csv_bytes = result.to_csv(index=False, sep=';', decimal=',').encode("utf-8")
        st.download_button("⬇️ Baixar CSV", data=csv_bytes, 
                         file_name="reservatorios_tabela_diaria.csv", 
                         mime="text/csv")

        st.caption(
            "As colunas com datas no cabeçalho exibem o **Nível** nas **duas datas mais recentes**. "
            "• **Capacidade Total (m³)** = Volume (dia atual) ÷ (Percentual (dia atual) ÷ 100). "
            "• **Variação do Nível** = Nível (data atual) − Nível (data anterior)."
        )

except Exception as e:
    st.error(f"Ocorreu um erro ao processar os dados: {str(e)}")
    import traceback
    with st.expander("Detalhes do erro"):
        st.code(traceback.format_exc())
