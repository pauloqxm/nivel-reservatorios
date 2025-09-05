import re
import math
import unicodedata
import pandas as pd
import streamlit as st
import altair as alt
from io import BytesIO

st.set_page_config(page_title="Reservatórios – Tabela diária", layout="wide")

# Altair: evitar limite de linhas
alt.data_transformers.disable_max_rows()

# ==========================
# Configuração
# ==========================
SHEETS_URL = "https://docs.google.com/spreadsheets/d/1zZ0RCyYj-AzA_dhWzxRziDWjgforbaH7WIoSEd2EKdk/edit?gid=1305065127#gid=1305065127"

# ==========================
# Utilitários
# ==========================
@st.cache_data(ttl=900)
def google_sheets_to_csv_url(url: str) -> str:
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
    aliases = set(list(aliases))
    normalized = {col: strip_accents_lower(col) for col in df.columns}
    for col, norm in normalized.items():
        if norm in aliases:
            return col
    for col, norm in normalized.items():
        if any(alias in norm for alias in aliases):
            return col
    return None

# ==========================
# Núcleo de cálculo
# ==========================
def compute_table_global_dates(
    df_raw: pd.DataFrame,
    forced_prev_date: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]:
    def last_scalar_on_date(dfr: pd.DataFrame, date_col: str, target_date, value_col: str) -> float:
        if pd.isna(target_date):
            return math.nan
        ddates = pd.to_datetime(dfr[date_col], errors="coerce").dt.normalize()
        sel = dfr.loc[ddates == pd.Timestamp(target_date).normalize(), value_col]
        if sel.empty:
            return math.nan
        sel = pd.to_numeric(sel, errors="coerce").dropna()
        if sel.empty:
            return math.nan
        return float(sel.iloc[-1])

    df = df_raw.copy()

    col_reservatorio = find_column(df, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    col_cota_sangria = find_column(df, {"cota sangria", "cota de sangria", "cota_sangria", "cota excedencia"})
    col_data           = find_column(df, {"data", "dt", "dia"})
    col_volume         = find_column(df, {"volume", "vol"})
    col_percentual     = find_column(df, {"percentual", "perc", "percentual (%)", "volume (%)"})
    col_nivel          = find_column(df, {"nivel", "nível", "cota", "altura"})

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
        raise ValueError("Não foi possível identificar as colunas: " + ", ".join(missing))

    df[col_data]           = df[col_data].apply(to_datetime_any)
    df[col_volume]         = df[col_volume].apply(to_number)
    df[col_percentual]     = df[col_percentual].apply(to_number)
    df[col_nivel]          = df[col_nivel].apply(to_number)
    df[col_cota_sangria] = df[col_cota_sangria].apply(to_number)
    df = df.dropna(subset=[col_data])

    unique_dates = pd.to_datetime(df[col_data].dropna().unique(), errors="coerce")
    unique_dates = pd.Series(unique_dates).dropna().sort_values().unique().tolist()
    if not unique_dates:
        return pd.DataFrame(), pd.NaT, pd.NaT, []

    data_atual = unique_dates[-1]
    prev_candidates = [d for d in unique_dates if d < data_atual]
    prev_options_desc = prev_candidates[::-1]

    if forced_prev_date and forced_prev_date in prev_candidates:
        data_anterior = forced_prev_date
    else:
        data_anterior = prev_candidates[-1] if prev_candidates else pd.NaT

    col_atual_label    = data_atual.strftime("%d/%m/%Y") if pd.notna(data_atual) else "Data Atual"
    col_anterior_label = data_anterior.strftime("%d/%m/%Y") if pd.notna(data_anterior) else "Data Anterior"

    rows = []
    for res, dfr in df.groupby(col_reservatorio, dropna=True):
        nivel_atual    = last_scalar_on_date(dfr, col_data, data_atual,    col_nivel)
        nivel_anterior = last_scalar_on_date(dfr, col_data, data_anterior, col_nivel) if pd.notna(data_anterior) else math.nan

        vol_atual      = last_scalar_on_date(dfr, col_data, data_atual,    col_volume)
        vol_anterior   = last_scalar_on_date(dfr, col_data, data_anterior, col_volume) if pd.notna(data_anterior) else math.nan
        perc_atual     = last_scalar_on_date(dfr, col_data, data_atual, col_percentual)

        cap_total = vol_atual / (perc_atual / 100.0) if (pd.notna(vol_atual) and pd.notna(perc_atual) and perc_atual != 0) else math.nan
        variacao_nivel  = (nivel_atual - nivel_anterior) if (pd.notna(nivel_atual) and pd.notna(nivel_anterior)) else math.nan
        variacao_volume = (vol_atual   - vol_anterior)  if (pd.notna(vol_atual)  and pd.notna(vol_anterior))  else math.nan

        cota_atual = last_scalar_on_date(dfr, col_data, data_atual, col_cota_sangria)
        if pd.notna(cota_atual):
            cota_sangria_val = cota_atual
        else:
            cota_hist = pd.to_numeric(dfr[col_cota_sangria], errors="coerce").dropna()
            cota_sangria_val = float(cota_hist.iloc[-1]) if not cota_hist.empty else math.nan
            
        verter_val = (cota_sangria_val - nivel_atual) if pd.notna(cota_sangria_val) and pd.notna(nivel_atual) else math.nan
        
        rows.append({
            "Reservatório": res,
            "Capacidade Total (m³)": cap_total,
            "Cota Sangria": cota_sangria_val,
            col_anterior_label: nivel_anterior,
            col_atual_label:     nivel_atual,
            "Variação do Nível": variacao_nivel,
            "Variação do Volume": variacao_volume,  # m³ (3 casas na exibição)
            "Volume": vol_atual,
            "Percentual": perc_atual,
            "Verter": verter_val,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        order = [
            "Reservatório",
            "Capacidade Total (m³)",
            "Cota Sangria",
            col_anterior_label, col_atual_label,
            "Variação do Nível",
            "Variação do Volume",
            "Volume", "Percentual", "Verter"
        ]
        out = out.reindex(columns=order).sort_values("Reservatório").reset_index(drop=True)

    return out, data_anterior, data_atual, prev_options_desc

# ==========================
# Formatação e renderização da tabela (HTML mesclado)
# ==========================
def format_ptbr(num, casas=2, inteiro=False):
    if pd.isna(num):
        return ""
    if inteiro:
        s = f"{num:,.0f}"
    else:
        s = f"{num:,.{casas}f}"
    return s.replace(",", "temp").replace(".", ",").replace("temp", ".")

def format_pct_br(num, casas=2):
    s = format_ptbr(num, casas=casas)
    return (s + "%") if s != "" else ""

def format_m3(num, casas=2):
    s = format_ptbr(num, casas=casas)
    return (s + " m³") if s != "" else ""

def var_icon_html(v):
    if pd.isna(v):
        return ""
    val = format_ptbr(v, casas=2)
    if v > 0:
        return f'{val} <span style="color:#2563eb">▲</span>'
    if v < 0:
        return f'{val} <span style="color:#dc2626">▼</span>'
    return f"{val} —"
    
def render_table_with_group_headers(
    df: pd.DataFrame,
    prev_label: str,
    curr_label: str,
    volume_group_label: str,
    cota_group_label: str = "Cota (m)",
    prev_date_value: pd.Timestamp = None  # Adicionado parâmetro para a data anterior
):
    css = """
    <style>
    table.cota-table {width: 100%; border-collapse: collapse; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 14px;}
    table.cota-table th, table.cota-table td {border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right;}
    table.cota-table th {background: #d1fae5; font-weight: 600; color: #111827; text-align: center;}
    table.cota-table td:first-child, table.cota-table th:first-child {text-align: left;}
    .group-head {text-align: center; background: #a7f3d0;}
    table.cota-table tr:nth-child(even) {background-color: #f7fee7;}
    .date-header {cursor: pointer; text-decoration: underline; color: #2563eb;}
    .date-header:hover {color: #1e40af;}
    </style>
    """
    html = [css, '<table class="cota-table">', "<thead>"]

    # Linha 1
    html.append("<tr>")
    html.append('<th rowspan="2">Reservatório</th>')
    html.append('<th rowspan="2">Capacidade Total (m³)</th>')
    html.append('<th rowspan="2">Cota Sangria</th>')
    html.append(f'<th class="group-head" colspan="2">{cota_group_label}</th>')
    html.append('<th rowspan="2">Variação do Nível</th>')
    html.append('<th rowspan="2">Variação do Volume</th>')
    html.append(f'<th class="group-head" colspan="2">{volume_group_label}</th>')
    html.append('<th rowspan="2">Verter (m)</th>')
    html.append("</tr>")

    # Linha 2 - MODIFICADO: cabeçalho da data anterior é clicável
    html.append("<tr>")
    
    # Cabeçalho da data anterior clicável
    prev_date_str = prev_date_value.strftime('%Y-%m-%d') if prev_date_value else ''
    html.append(f'<th class="date-header" onclick="alterarDataAnterior(\'{prev_date_str}\')">{prev_label} 📅</th>')
    
    html.append(f"<th>{curr_label}</th>")
    html.append("<th>Volume</th>")
    html.append("<th>Percentual (%)</th>")
    html.append("</tr>")
    html.append("</thead>")

    # Corpo
    html.append("<tbody>")
    for _, row in df.iterrows():
        html.append("<tr>")
        html.append(f"<td>{row['Reservatório']}</td>")
        html.append(f"<td>{format_ptbr(row['Capacidade Total (m³)'], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row['Cota Sangria'], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row[prev_label], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row[curr_label], casas=2)}</td>")
        html.append(f"<td>{var_icon_html(row['Variação do Nível'])}</td>")
        html.append(f"<td>{format_m3(row['Variação do Volume'], casas=3)}</td>")
        html.append(f"<td>{format_ptbr(row['Volume'], casas=2)}</td>")
        html.append(f"<td>{format_pct_br(row['Percentual'], casas=2)}</td>")
        html.append(f"<td>{format_ptbr(row['Verter'], casas=2)}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    # JavaScript para alterar a data anterior
    html.append("""
    <script>
    function alterarDataAnterior(currentDate) {
        // Criar um formulário para enviar a data via query params
        const form = document.createElement('form');
        form.method = 'GET';
        
        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'prev';
        input.value = currentDate;
        
        form.appendChild(input);
        document.body.appendChild(form);
        form.submit();
    }
    </script>
    """)

    return "\n".join(html)

# ==========================
# UI (sem barra lateral) + Calendário + Tabela + Gráficos
# ==========================
st.title("📊 Tabela diária de Reservatórios")

try:
    df_raw = load_data_from_url(SHEETS_URL)
    
    # Filtro por reservatório
    col_res_guess = find_column(df_raw, {"reservatorio", "reservatório", "acude", "açude", "nome"})
    if col_res_guess:
        reservatorios = sorted(x for x in df_raw[col_res_guess].dropna().unique() if x)
        
        # Usar session_state para gerenciar a seleção
        if 'selected_reservoirs' not in st.session_state:
            st.session_state.selected_reservoirs = ["Todos"]
        
        # Container para organizar os controles
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Multiselect para seleção - MOSTRAR TODOS OS NOMES QUANDO "Todos" ESTIVER SELECIONADO
            if "Todos" in st.session_state.selected_reservoirs:
                # Mostrar todos os nomes selecionados quando "Todos" está ativo
                sel = st.multiselect(
                    "Filtrar reservatórios (opcional)", 
                    reservatorios, 
                    default=reservatorios,  # TODOS SELECIONADOS
                    placeholder="Selecione…",
                    help="Todos os reservatórios selecionados. Desselecione individualmente para remover."
                )
            else:
                # Mostrar seleção normal quando não está em "Todos"
                sel = st.multiselect(
                    "Filtrar reservatórios (opcional)", 
                    ["Todos"] + reservatorios, 
                    default=st.session_state.selected_reservoirs,
                    placeholder="Selecione…",
                    help="Selecione 'Todos' para mostrar todos os reservatórios, ou selecione/desselecione individualmente"
                )
        
        with col2:
            # Botões de ação rápida
            st.write("Ações rápidas:")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Todos", use_container_width=True):
                    # Seleciona TODOS os nomes (como na imagem)
                    st.session_state.selected_reservoirs = reservatorios
                    st.rerun()
            with col_btn2:
                if st.button("Limpar", use_container_width=True):
                    st.session_state.selected_reservoirs = []
                    st.rerun()
        
        # Atualizar session_state se a seleção mudou
        if sel != st.session_state.selected_reservoirs:
            st.session_state.selected_reservoirs = sel
            st.rerun()
        
        # Lógica de filtragem
        if not st.session_state.selected_reservoirs:
            df_filtered = df_raw.head(0)
            st.info("Nenhum reservatório selecionado. Selecione 'Todos' ou reservatórios específicos para visualizar os dados.")
            st.stop()
        elif len(st.session_state.selected_reservoirs) == len(reservatorios):
            # Se TODOS os reservatórios estão selecionados, mostrar tudo
            df_filtered = df_raw
            st.info("Mostrando todos os reservatórios")
        else:
            # Se há seleções específicas
            df_filtered = df_raw[df_raw[col_res_guess].isin(st.session_state.selected_reservoirs)]
            st.info(f"Mostrando {len(st.session_state.selected_reservoirs)} reservatório(s) selecionado(s)")
            
        # Mostrar status da seleção
        if st.session_state.selected_reservoirs:
            if len(st.session_state.selected_reservoirs) == len(reservatorios):
                st.caption("Selecionados: Todos os reservatórios")
            else:
                st.caption(f"Selecionados: {', '.join(st.session_state.selected_reservoirs)}")
            
    else:
        df_filtered = df_raw
        st.warning("Não foi possível identificar a coluna de Reservatório.")

    # Query param para data anterior
    q = st.query_params
    forced_prev = None
    if "prev" in q and q["prev"]:
        try:
            forced_prev = pd.to_datetime(q["prev"], errors="coerce").normalize()
        except Exception:
            forced_prev = None

    # Processa
    with st.spinner("Processando dados..."):
        result, dprev, dcurr, prev_options_desc = compute_table_global_dates(df_filtered, forced_prev_date=forced_prev)

    st.subheader("Resultado")

    # Tabela (HTML mesclado na página) - MODIFICADO
    if result.empty:
        st.info("Nenhum dado encontrado para as datas selecionadas.")
        st.stop()

    prev_label = dprev.strftime("%d/%m/%Y") if pd.notna(dprev) else "Data Anterior"
    curr_label = dcurr.strftime("%d/%m/%Y") if pd.notna(dcurr) else "Data Atual"
    volume_group_label = f"Volume ({curr_label})"

    html_table_string = render_table_with_group_headers(
        result,
        prev_label=prev_label,
        curr_label=curr_label,
        volume_group_label=volume_group_label,
        cota_group_label="Cota (m)",
        prev_date_value=dprev  # Passar a data anterior para o cabeçalho
    )
    st.markdown(html_table_string, unsafe_allow_html=True)

    # POPOVER PARA SELECIONAR DATA - agora aparece quando clica na coluna
    if st.query_params.get("prev"):
        # Se há um parâmetro 'prev' na URL, mostrar o popover de seleção de data
        with st.popover("📅 Selecionar data anterior", open=True):
            st.markdown("**Escolha a data anterior**")
            
            if prev_options_desc:
                min_prev = prev_options_desc[-1].date()
                max_prev = prev_options_desc[0].date()
            else:
                min_prev = None
                max_prev = None
            
            default_date = dprev.date() if pd.notna(dprev) else (max_prev or pd.Timestamp.today().date())
            date_sel = st.date_input(
                "Data anterior",
                value=default_date,
                min_value=min_prev,
                max_value=(dcurr - pd.Timedelta(days=1)).date() if pd.notna(dcurr) else None,
                format="DD/MM/YYYY"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Aplicar", type="primary"):
                    st.query_params.update({"prev": pd.Timestamp(date_sel).strftime("%Y-%m-%d")})
                    st.rerun()
            with col2:
                if st.button("Cancelar"):
                    # Limpar o parâmetro prev
                    if "prev" in st.query_params:
                        del st.query_params["prev"]
                    st.rerun()

    # ===== Botões de exportação =====
    col1, col2 = st.columns([1, 1])

    with col1:
        # CSV (formatado)
        csv_bytes = result.to_csv(index=False, sep=';', decimal=',').encode("utf-8")
        st.download_button("⬇️ Baixar CSV (formatado)", data=csv_bytes,
                           file_name="reservatorios_tabela_diaria.csv", mime="text/csv", use_container_width=True)
    with col2:
        # HTML da Tabela
        st.download_button(
            label="🌐 Baixar HTML — Tabela",
            data=html_table_string.encode("utf-8"),
            file_name="tabela_reservatorios.html",
            mime="text/html",
            use_container_width=True
        )


    # ==========================
    # GRÁFICOS (Altair) — TODOS com altura dinâmica (todas as linhas)
    # ==========================
    st.markdown("---")
    st.markdown("### 📈 Visualizações")

    # Δ Nível (m)
    if "Variação do Nível" in result.columns:
        df_var_nivel = result[["Reservatório", "Variação do Nível"]].dropna()
        if not df_var_nivel.empty:
            h1 = max(220, 24 * len(df_var_nivel))
            var_nivel_chart = (
                alt.Chart(df_var_nivel)
                .mark_bar()
                .encode(
                    y=alt.Y("Reservatório:N", sort=alt.EncodingSortField(field="Variação do Nível", op="min", order='ascending'), title=None),
                    x=alt.X("Variação do Nível:Q", title="Δ nível (m)"),
                    color=alt.condition("datum['Variação do Nível'] > 0",
                                        alt.value("#2563eb"),  # azul
                                        alt.value("#dc2626")), # vermelho
                    tooltip=[
                        alt.Tooltip("Reservatório:N"),
                        alt.Tooltip("Variação do Nível:Q", format=".2f"),
                    ],
                )
                .properties(height=h1, title="Δ Nível (m)")
            )
            st.altair_chart(var_nivel_chart.interactive(), use_container_width=True)

    # Volume vs Capacidade (bullet)
    cap_cols = ["Reservatório", "Capacidade Total (m³)", "Volume", "Percentual"]
    cap_df = result[cap_cols].dropna(subset=["Capacidade Total (m³)", "Volume"])
    if not cap_df.empty:
        h2 = max(240, 26 * len(cap_df))
        back = (
            alt.Chart(cap_df)
            .mark_bar(size=16, opacity=0.35, color="#94a3b8")
            .encode(
                y=alt.Y("Reservatório:N", sort=alt.EncodingSortField(field="Volume", op="min", order='descending'), title=None),
                x=alt.X("Capacidade Total (m³):Q", title="m³"),
                tooltip=[
                    alt.Tooltip("Reservatório:N"),
                    alt.Tooltip("Capacidade Total (m³):Q", format=".2f"),
                ],
            )
        )
        front = (
            alt.Chart(cap_df)
            .mark_bar(size=10, color="#2563eb")
            .encode(
                y=alt.Y("Reservatório:N", sort=alt.EncodingSortField(field="Volume", op="min", order='descending')),
                x=alt.X("Volume:Q"),
                tooltip=[
                    alt.Tooltip("Reservatório:N"),
                    alt.Tooltip("Volume:Q", format=".2f"),
                    alt.Tooltip("Percentual:Q", format=".2f", title="Percentual (%)"),
                ],
            )
        )
        bullet = (back + front).properties(height=h2, title="Volume atual vs Capacidade total (m³)")
        st.altair_chart(bullet.interactive(), use_container_width=True)

    # Δ Volume (m³) — 3 casas
    if "Variação do Volume" in result.columns:
        df_var_vol = result[["Reservatório", "Variação do Volume"]].dropna()
        if not df_var_vol.empty:
            h3 = max(220, 24 * len(df_var_vol))
            var_vol_chart = (
                alt.Chart(df_var_vol)
                .mark_bar()
                .encode(
                    y=alt.Y("Reservatório:N", sort=alt.EncodingSortField(field="Variação do Volume", op="min", order='ascending'), title=None),
                    x=alt.X("Variação do Volume:Q", title="Δ volume (m³)"),
                    color=alt.condition("datum['Variação do Volume'] > 0",
                                        alt.value("#2563eb"),
                                        alt.value("#dc2626")),
                    tooltip=[
                        alt.Tooltip("Reservatório:N"),
                        alt.Tooltip("Variação do Volume:Q", format=".3f"),
                    ],
                )
                .properties(height=h3, title="Δ Volume (m³)")
            )
            st.altair_chart(var_vol_chart.interactive(), use_container_width=True)
            
    # Verter (m) - Adicionado
    if "Verter" in result.columns:
        df_verter = result[["Reservatório", "Verter"]].dropna()
        if not df_verter.empty:
            h4 = max(220, 24 * len(df_verter))
            verter_chart = (
                alt.Chart(df_verter)
                .mark_bar()
                .encode(
                    y=alt.Y("Reservatório:N", sort=alt.EncodingSortField(field="Verter", op="min", order='descending'), title=None),
                    x=alt.X("Verter:Q", title="Verter (m)"),
                    color=alt.condition("datum['Verter'] <= 0",
                                        alt.value("#34d399"), # Verde (verter)
                                        alt.value("#facc15")), # Amarelo (não verter)
                    tooltip=[
                        alt.Tooltip("Reservatório:N"),
                        alt.Tooltip("Verter:Q", format=".2f"),
                    ],
                )
                .properties(height=h4, title="Distância para Verter (m)")
            )
            st.altair_chart(verter_chart.interactive(), use_container_width=True)

    st.caption("As alturas dos gráficos se ajustam automaticamente para exibir todas as linhas. ")

except Exception as e:
    st.error(f"Ocorreu um erro ao processar os dados: {str(e)}")
    import traceback
    with st.expander("Detalhes do erro"):
        st.code(traceback.format_exc())
