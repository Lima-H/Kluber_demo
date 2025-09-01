import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import timedelta



sns.set_palette("pastel")
st.set_page_config(page_title="Predição de Compras", layout="wide")
st.title("📊 Predição de Compras - 30, 60 e 90 dias")

with st.expander("ℹ️ Sobre o modelo", expanded=True):
    st.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 16px; border-radius: 8px;">
        <b>O modelo foi treinado apenas com dados anteriores a 2024.</b> Os dados de 2024 foram reservados exclusivamente para testar a assertividade das previsões.<br><br>
        Para prever se um cliente irá realizar uma nova compra em 30, 60 ou 90 dias, criamos diversas <b>features</b> a partir do histórico de compras, como:
        <ul>
            <li>Recorrência e recência histórica de compras</li>
            <li>Valor médio gasto</li>
            <li>Mês e período do mês da compra</li>
            <li>Entre outras variáveis comportamentais</li>
        </ul>
        Todas essas informações são utilizadas para aumentar a precisão das previsões apresentadas abaixo.
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================
# 1️⃣ Carregar os dados
# =====================
df = pd.read_csv("dados_predicoes.csv")
df["data_compra"] = pd.to_datetime(df["data_compra"])

# Última compra de cada cliente
df_clientes = df.sort_values(["cliente_id", "data_compra"], ascending=[True, False])
 
# Pegando a penúltima compra do cliente
df_aux = pd.DataFrame(columns=df_clientes.columns)
for id in df_clientes["cliente_id"].unique():
    aux = df_clientes.loc[df_clientes["cliente_id"] == id]
    aux.reset_index(drop=True, inplace=True)
    try: 
        aux = aux.iloc[1]
    except:
        aux = aux.iloc[0]
    df_aux = pd.concat([df_aux, aux.to_frame().T], ignore_index=True)

df_clientes = df_aux.copy()

# Pegando a última compra do cliente
# df_clientes = df_clientes.drop_duplicates(subset="cliente_id", keep="first").copy()

# ---- Modelos 180/360 ----
df2 = pd.read_csv("dados_predicoes2.csv")
df2["data_compra"] = pd.to_datetime(df2["data_compra"])

df2_clientes = df2.sort_values(["cliente_id", "data_compra"], ascending=[True, False])
df2_aux = pd.DataFrame(columns=df2_clientes.columns)
for id in df2_clientes["cliente_id"].unique():
    aux = df2_clientes.loc[df2_clientes["cliente_id"] == id]
    aux.reset_index(drop=True, inplace=True)
    try:
        aux = aux.iloc[1]
    except:
        aux = aux.iloc[0]
    df2_aux = pd.concat([df2_aux, aux.to_frame().T], ignore_index=True)

df2_clientes = df2_aux.copy()

# =====================
# 2️⃣ Métricas e Matrizes de Confusão
# =====================
with st.expander("📈 Resultados da Assertividade do Modelo", expanded=False):
    periodos = [("30 dias", "comprou_30d", "prob_30d"),
                ("60 dias", "comprou_60d", "prob_60d"),
                ("90 dias", "comprou_90d", "prob_90d")]

    col1, col2, col3 = st.columns(3)

    for (titulo, y_col, p_col), col in zip(periodos, [col1, col2, col3]):
        y_true = df[y_col]
        y_pred = (df[p_col] >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        with col:
            st.metric(f"Acurácia {titulo}", f"{acc:.2%}")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="BuGn", ax=ax)
            ax.set_xlabel("Previsto")
            ax.set_ylabel("Real")
            st.pyplot(fig)

    st.subheader("Modelos de 180 e 360 dias")
    periodos2 = [
        ("180 dias", "comprou_180d", "prob_180d"),
        ("360 dias", "comprou_360d", "prob_360d"),
    ]

    col4, col5, col6 = st.columns([1, 1, 0.5])

    for (titulo, y_col, p_col), col in zip(periodos2, [col4, col5]):
        y_true = df2[y_col]
        y_pred = (df2[p_col] >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        with col:
            st.metric(f"Acurácia {titulo}", f"{acc:.2%}")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="BuGn", ax=ax)
            ax.set_xlabel("Previsto")
            ax.set_ylabel("Real")
            st.pyplot(fig)

# =====================
# 3️⃣ Exploração de Clientes
# =====================
# =====================
# 3️⃣ Exploração de Clientes
# =====================
st.header("Simulação de Clientes")

# ---- Escolha do conjunto de modelos ----
conjunto = st.radio(
    "Escolha o horizonte de previsão",
    ["30-60-90 dias", "180-360 dias"]
)

if conjunto == "30-60-90 dias":
    df_base = df
    df_clientes_base = df_clientes
    periodos_disp = ["prob_30d", "prob_60d", "prob_90d"]
elif conjunto == "180-360 dias":
    df_base = df2
    df_clientes_base = df2_clientes
    periodos_disp = ["prob_180d", "prob_360d"]

# ---- Filtro geral ----
col1, col2 = st.columns(2)

with col1:
    periodo_filtro = st.selectbox("Período para filtrar", periodos_disp)
    prob_min = st.slider("Probabilidade mínima", 0.0, 1.0, 0.5, 0.01)
    df_filtrado = df_clientes_base[df_clientes_base[periodo_filtro] >= prob_min]
    st.write(f"{len(df_filtrado)} clientes encontrados")

    # Mostrar apenas colunas disponíveis naquele conjunto
    cols_para_mostrar = ["cliente_id", "Valor_Médio"] + periodos_disp
    st.dataframe(df_filtrado[cols_para_mostrar].sort_values(periodo_filtro, ascending=False))

# ---- Busca individual ----
with col2:
    cliente_id_str = st.text_input("Buscar cliente pelo ID")
    if cliente_id_str:
        cliente_id_str = cliente_id_str.strip()
        # tenta converter apenas o input para int (capture só erro de conversão aqui)
        try:
            cliente_id = int(cliente_id_str)
        except ValueError:
            st.error("Digite um número válido para o ID do cliente.")
        else:
            cliente_info = df_clientes_base[df_clientes_base["cliente_id"] == cliente_id]
            if cliente_info.empty:
                st.warning("Cliente não encontrado.")
            else:
                st.subheader(f"Cliente {cliente_id}")

                # Métrica de probabilidade
                prob_val = cliente_info[periodo_filtro].values[0]
                st.metric(f"Probabilidade ({periodo_filtro})", f"{prob_val:.2%}")

                # Histórico completo do cliente (cópia para segurança)
                historico = df_base[df_base["cliente_id"] == cliente_id].copy()

                total_compras = historico.shape[0]
                preco_medio = historico["Valor_Médio"].mean() if total_compras > 0 else 0.0
                data_ultima_compra = cliente_info["data_compra"].iloc[0]

                st.write(f"**Preço médio dos pedidos:** R$ {preco_medio:.2f}")
                st.write(f"**Data compra considerada para previsão:** {pd.to_datetime(data_ultima_compra).strftime('%d/%m/%Y')}")

                # Função para explodir produtos (usa split + explode corretamente)
                def separar_produtos_df(df, coluna="Produto"):
                    df = df.copy()
                    # evita NaNs e garante lista
                    df[coluna] = df[coluna].fillna("").astype(str).str.split(",")
                    df = df.explode(coluna)
                    df[coluna] = df[coluna].str.strip()
                    # remove strings vazias que vieram de NaN ou field vazio
                    df = df[df[coluna] != ""]
                    return df

                # -----------------------------
                # Heatmap estilo GitHub: Produto x Mês
                # -----------------------------
                try:
                    ultima_data = pd.to_datetime(cliente_info["data_compra"].values[0])
                except Exception:
                    ultima_data = pd.to_datetime(data_ultima_compra)

                # extrai N dias do periodo_filtro (ex: "prob_30d" -> 30)
                try:
                    periodo = int(periodo_filtro.split('_')[1][:-1])
                except Exception:
                    periodo = 30  # fallback seguro

                inicio_janela = ultima_data - timedelta(days=periodo)

                # Filtra histórico na janela selecionada
                hist_filtro = historico.copy()
                hist_filtro["data_compra"] = pd.to_datetime(hist_filtro["data_compra"])
                hist_filtro = hist_filtro[hist_filtro["data_compra"] >= inicio_janela]

                # Explodir produtos corretamente
                hist_filtro = separar_produtos_df(hist_filtro, "Produto")

                if hist_filtro.empty:
                    st.write(f"**Produtos por mês (últimos {periodo} dias)**")
                    st.info("Sem compras do cliente na janela selecionada.")
                else:
                    # Mês/Ano como Period
                    hist_filtro["mes_ano"] = hist_filtro["data_compra"].dt.to_period("M")

                    # faixa completa de meses (mes_inicio .. mes_fim)
                    mes_inicio = inicio_janela.to_period("M")
                    mes_fim = ultima_data.to_period("M")
                    faixa_meses = pd.period_range(mes_inicio, mes_fim, freq="M")

                    # marca compra
                    hist_filtro["comprou"] = 1

                    # pivot (Produto x Mes) sem duplicação de mesmo produto/mes
                    tabela_produtos = (
                        hist_filtro
                        .drop_duplicates(subset=["Produto", "mes_ano"])
                        .pivot_table(
                            index="Produto",
                            columns="mes_ano",
                            values="comprou",
                            aggfunc="max",
                            fill_value=0
                        )
                    )

                    # garante colunas para todos os meses da janela (preenchendo com 0)
                    # converte colunas para PeriodIndex antes de reindexar se necessário
                    try:
                        tabela_produtos = tabela_produtos.reindex(columns=faixa_meses, fill_value=0)
                    except Exception:
                        # fallback: converter colunas para string e reindex por strings
                        tabela_produtos.columns = tabela_produtos.columns.astype(str)
                        faixa_meses_str = [str(p) for p in faixa_meses]
                        tabela_produtos = tabela_produtos.reindex(columns=faixa_meses_str, fill_value=0)

                    # ordena meses e produtos mais ativos no topo
                    tabela_produtos = tabela_produtos.sort_index(axis=1)
                    tabela_produtos = tabela_produtos.loc[tabela_produtos.sum(axis=1).sort_values(ascending=False).index]

                    # prepara para plot
                    tabela_plot = tabela_produtos.copy()
                    tabela_plot.columns = tabela_plot.columns.astype(str)

                    st.write(f"**Produtos comprados por mês (janela: últimos {periodo} dias)**")

                    # Gera o heatmap (coloca try para capturar erro real se acontecer)
                    try:
                        n_rows, n_cols = tabela_plot.shape
                        fig_w = min(16, max(6, 0.8 * n_cols + 3))
                        fig_h = min(18, max(4, 0.35 * n_rows + 1))
                        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

                        sns.heatmap(
                            tabela_plot.astype(int),
                            cmap="Greens",
                            cbar=False,
                            linewidths=0.5,
                            linecolor="lightgrey",
                            ax=ax
                        )
                        ax.set_xlabel("Mês")
                        ax.set_ylabel("Produto")
                        ax.set_title(f"Produtos comprados por mês (últimos {periodo} dias)")
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

                        st.pyplot(fig)
                    except Exception as e_plot:
                        st.error("Erro ao gerar o gráfico de heatmap.")
                        st.exception(e_plot)
                
                # Remove colunas específicas do DataFrame antes de exibir
                historico_filtrado = historico.drop(columns=["Produto", "Valor_Médio"], errors="ignore")

                # Mantém o histórico original
                st.write(f"**Histórico do cliente**")
                st.dataframe(historico_filtrado)