# app.py
import re
import math
import unicodedata
import pandas as pd
import streamlit as st

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
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

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
    """Converte datas em vários formatos, retornando Timestamp normalizado (sem hora) ou NaT."""
    try:
        ts = pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT
    return ts.normalize() if pd.notna(ts) else pd.NaT

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
# Núcleo de cálculo
# ==========================
def compute_table_global_dates(df_raw: pd.DataFrame, forced_prev_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Retorna (df, data_anterior, data_atual). Cabeçalhos de datas serão dd/mm/aaaa.
    Se forced_prev_date for passado, usa-o como 'data anterior' (se existir).
    """
    def last_scalar_on_date(dfr: pd.DataFrame, date_col: str, target_date, value_col: str) -> float:
        """Último valor numérico (float) para a data exata target_date. Sempre escalar ou NaN."""
        if pd.isna(target_date):
            return math.nan
        # Garante comparação só por data (normalizada)
        ddates = pd.to_datetime(dfr[date_col], errors="coerce").dt.normalize()
        sel = dfr.loc[ddates == pd.Timestamp(target_date).normalize(), value_col]
        if sel.empty:
            return math.nan
        sel = pd.to_numeric(sel, errors="coerce").dropna()
        if sel.empty:
            return math.nan
        return float(sel.iloc[-1])

    df = df_raw.copy()

    # Mapear colunas
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
    if missing:
        raise ValueError(
            "Não foi possível identificar as colunas na planilha. "
            f"Faltando: {', '.join(missing)}. Ajuste os aliases no código ou renomeie na planilha."
        )

    # Conversões
    df[col_data]         = df[col_data].apply(to_datetime_any)
    df[col_volume]       = df[col_volume].apply(to_number)
    df[col_percentual]   = df[col_percentual].apply(to_number)
    df[col_nivel]        = df[col_nivel].apply(to_number)
    df[col_cota_sangria] = df[col_cota_sangria].apply(to_number)
    df = df.dropna(subset=[col_data])

    # Datas disponíveis (normalizadas, ordenadas)
    unique_dates = pd.to_datetime(df[col_data].dropna().unique(), errors="coerce")
    unique_dates = pd.Series(unique_dates).dropna().sort_values().unique().tolist()
    if not unique_dates:
        return pd.DataFrame(), pd.NaT, pd.NaT

    data_atual = unique_dates[-1]
    # Candidatas de 'data anterior' (antes da atual)
    prev_candidates = [d for d in unique_dates if d < data_atual]
    if forced_prev_date and forced_prev_date in prev_candidates:
        data_anterior = forced_prev_date
    else:
        data_anterior = prev_candidates[-1] if prev_candidates else pd.NaT

    # Cabeçalhos com as datas (formatos dd/mm/aaaa)
    col_atual_label    = data_atual.strftime("%d/%m/%Y") if pd.notna(data_atual) else "Data Atual"
    col_anterior_label = data_anterior.strftime("%d/%m/%Y") if pd.notna(data_anterior) else "Data Anterior"

    rows = []
    for res, dfr in df.groupby(col_reservatorio, dropna=True):
        # Níveis nas duas datas
        nivel_atual    = last_scalar_on_date(dfr, col_data, data_atual,    col_nivel)
        nivel_anterior = last_scalar_on_date(dfr, col_data, data_anterior, col_nivel) if pd.notna(data_anterior) else math.nan

        # Capacidade Total a partir do dia atual
        vol_atual  = last_scalar_on_date(dfr, col_data, data_atual, col_volume)
        perc_atual = last_scalar_on_date(dfr, col_data, data_atual, col_percentual)
        cap_total = vol_atual / (perc_atual / 100.0) if (pd.notna(vol_atual) and pd.notna(perc_atual) and perc_atual != 0) else math.nan

        # Cota de sangria (preferir no dia atual; senão último histórico não-nulo)
        cota_atual = last_scalar_on_date(dfr, col_data, data_atual, col_cota_sangria)
        if pd.notna(cota_atual):
            cota_sangria_val = cota_atual
        else:
            cota_hist = pd.to_numeric(dfr[col_cota_sangria], errors="coerce").dropna()
            cota_sangria_val = float(cota_hist.iloc[-1]) if not cota_hist.empty else math.nan

        variacao = (nivel_atual - nivel_anterior) if (pd.notna(nivel_atual) and pd.notna(nivel_anterior)) else math.nan

        rows.append({
            "Reservatório": res,
            "Cota Sangria": cota_sangria_val,
            col_anterior_label: nivel_anterior,  # Nível (m) na data anterior
            col_atual_label:    nivel_atual,     # Nível (m) na data atual
            "Capacidade Total (m³)": cap_total,
            "Variação do Nível": variacao,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        order = ["Reservatório", "Cota Sangria", col_anterior_label, col_atual_label, "Capacidade Total (m³)", "Variação do Nível"]
        out = out.reindex(columns=order).sort_values("Reservatório").reset_index(drop=True)

    return out, data_anterior, data_atual

# ==========================
# Renderização com “célula mesclada” (HTML)
# ==========================
def format_ptbr(num, casas=2, inteiro=False):
    if pd.isna(num):
        return ""
    if inteiro:
        s = f"{num:,.0f}"
    else:
        s = f"{num:,.{casas}f}"
    # formatação pt-BR
    return s.replace(",", "temp").replace(".", ",").replace("temp", ".")

def render_table_with_group_header(df: pd.DataFrame, prev_label: str, curr_label: str, group_label="Cota (m)"):
    """Renderiza uma tabela HTML com cabeçalho mesclado sobre as duas colunas de data."""
    # nomes das colunas
    cols = list(df.columns)
    # índices das colunas de data
    i_prev = cols.index(prev_label)
    i_curr = cols.index(curr_label)

    # CSS simples para deixar bonito
    css = """
    <style>
    table.cota-table {width: 100%; border-collapse: collapse; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 14px;}
    table.cota-table th, table.cota-table td {border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right;}
    table.cota-table th {background: #f8fafc; font-weight: 600; color: #111827;}
    table.cota-table td:first-child, table.cota-table th:first-child {text-align: left;}
    table.cota-table td:nth-child(2), table.cota-table th:nth-child(2) {text-align: right;}
    .group-head {text-align: center; background: #eef2ff;}
    </style>
    """

    # Cabeçalho em duas linhas: mescla para “Cota (m)”
    # Colunas fixas à esquerda e direita têm rowspan=2
    left_fixed = ["Reservatório", "Cota Sangria"]
    right_fixed = ["Capacidade Total (m³)", "Variação do Nível"]

    html = [css, '<table class="cota-table">', "<thead>"]

    # Linha 1 do cabeçalho
    html.append("<tr>")
    html.append(f'<th rowspan="2">{left_fixed[0]}</th>')
    html.append(f'<th rowspan="2">{left_fixed[1]}</th>')
    html.append(f'<th class="group-head" colspan="2">{group_label}</th>')
    html.append(f'<th rowspan="2">{right_fixed[0]}</th>')
    html.append(f'<th rowspan="2">{right_fixed[1]}</th>')
    html.append("</tr>")

    # Linha 2 do cabeçalho (datas)
    html.append("<tr>")
    html.append(f"<th>{prev_label}</th>")
    html.append(f"<th>{curr_label}</th>")
    html.append("</tr>")
    html.append("</thead>")

    # Corpo
    html.append("<tbody>")
    for _, row in df.iterrows():
        html.append("<tr>")
        html.append(f"<td>{row[left_fixed[0]]}</td>")
        html.append(f"<td>{format_ptbr(row[left_fixed[1]], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row[prev_label], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row[curr_label], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row[right_fixed[0]], inteiro=True)}</td>")
        html.append(f"<td>{format_ptbr(row[right_fixed[1]], casas=2)}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    st.markdown("\n".join(html), unsafe_allow_html=True)

# ==========================
# UI
# ==========================
st.title("📊 Tabela diária de Reservatórios")

with st.sidebar:
    st.markdown("### Fonte dos dados")
    mode = st.radio("Escolha a fonte", ("Google Sheets (link padrão)", "Enviar CSV (arquivo local)"), index=0)
    if mode == "Google Sheets (link padrão)":
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

    with st.expander("Visualizar dados brutos"):
        st.dataframe(df_raw.head(), use_container_width=True)

    # Descobrir todas as datas disponíveis (normalizadas) para o seletor
    col_data_guess = find_column(df_raw, {"data", "dt", "dia"})
    date_options = []
    if col_data_guess:
        dnorm = pd.to_datetime(df_raw[col_data_guess], errors="coerce").dropna().dt.normalize()
        if not dnorm.empty:
            udates = sorted(dnorm.unique())
            data_atual_default = udates[-1]
            prev_cands = [d for d in udates if d < data_atual_default]
            # opções para o seletor (apenas anteriores à atual)
            opts = prev_cands[::-1]  # mais recentes primeiro
            date_options = [pd.Timestamp(d) for d in opts]

    forced_prev = None
    if date_options:
        forced_prev_str = st.sidebar.selectbox(
            "Selecionar data anterior",
            [d.strftime("%d/%m/%Y") for d in date_options],
            index=0,
            help="Escolha outra data para comparar com o dia atual."
        )
        forced_prev = pd.to_datetime(forced_prev_str, dayfirst=True)

    # Filtro opcional por reservatório
    col_res_guess = find_column(df_raw, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    if col_res_guess:
        reservatorios = sorted(x for x in df_raw[col_res_guess].dropna().unique() if x)
        sel = st.multiselect("Filtrar reservatórios (opcional)", reservatorios, [])
        df_filtered = df_raw[df_raw[col_res_guess].isin(sel)] if sel else df_raw
    else:
        df_filtered = df_raw
        st.warning("Não foi possível identificar a coluna de Reservatório.")

    # Processar
    with st.spinner("Processando dados..."):
        result, dprev, dcurr = compute_table_global_dates(df_filtered, forced_prev_date=forced_prev)

    st.subheader("Resultado")
    if result.empty:
        st.info("Nenhum dado encontrado para as datas selecionadas.")
    else:
        prev_label = dprev.strftime("%d/%m/%Y") if pd.notna(dprev) else "Data Anterior"
        curr_label = dcurr.strftime("%d/%m/%Y") if pd.notna(dcurr) else "Data Atual"

        # Render com “célula mesclada” Cota (m)
        render_table_with_group_header(result, prev_label, curr_label, group_label="Cota (m)")

        # Download CSV (dados crus)
        csv_bytes = result.to_csv(index=False, sep=';', decimal=',').encode("utf-8")
        st.download_button("⬇️ Baixar CSV", data=csv_bytes,
                           file_name="reservatorios_tabela_diaria.csv",
                           mime="text/csv")

        st.caption(
            "O cabeçalho **Cota (m)** agrupa os níveis medidos nas duas datas. "
            "Você pode ajustar a **data anterior** na barra lateral. "
            "• **Capacidade Total (m³)** = Volume (dia atual) ÷ (Percentual (dia atual) ÷ 100). "
            "• **Variação do Nível** = Nível (data atual) − Nível (data anterior)."
        )

except Exception as e:
    st.error(f"Ocorreu um erro ao processar os dados: {str(e)}")
    import traceback
    with st.expander("Detalhes do erro"):
        st.code(traceback.format_exc())
