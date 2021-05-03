"""
Start dateを基準にした株価の増加率をplot
比較基準の株価指標との変化を確認する
Usage:
    $ conda activate lightning
    $ streamlit run ./app.py
"""
import streamlit as st

import os
import gc
import glob
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set()

index_dir = "index_data"
stock_dir = "TOPIX100_data"


def load_df(d_dir, name):
    df = pd.read_csv(
        d_dir + "/" + str(name) + ".csv",
        encoding="SHIFT-JIS",
        sep="\t",
        parse_dates=["日付"],
        na_values=["-"],
        dtype="float",
    )
    return df


@st.cache(allow_output_mutation=True)
def get_stock_data(st_index):
    # 株価指標のデータロード
    return load_df(index_dir, st_index).sort_values(by="日付").reset_index(drop=True)


@st.cache(allow_output_mutation=True)
def get_stock_col_data(col):
    # 個別銘柄のデータロード
    stock_ids = [Path(csv).stem for csv in glob.glob(stock_dir + "/*csv")]
    df_stocks = None
    for s_id in stock_ids:
        df = load_df(stock_dir, s_id)
        df = df[["日付", col]].rename(columns={col: str(s_id)})
        if df_stocks is None:
            df_stocks = df
        else:
            df_stocks = pd.merge(df_stocks, df, on="日付", how="outer")
    return df_stocks.sort_values(by="日付").reset_index(drop=True)


def get_merge_data(st_index, col):
    # 株価指標のデータロード
    df_index = get_stock_data(st_index)
    df_index = df_index[["日付", col]].rename(columns={col: st_index})

    # 個別銘柄のデータロード
    df_stocks = get_stock_col_data(col)

    # 株価指標と個別銘柄マージ
    df_merge = pd.merge(df_index, df_stocks, on="日付", how="outer")

    return df_merge.sort_values(by="日付").reset_index(drop=True)


def main():
    # サイドバー
    st_index = st.sidebar.selectbox("比較基準の株価指標", ("nikkei225", "jpx400", "topix"))
    st_p_type = st.sidebar.selectbox("価格の種類", ("始値", "終値", "高値", "安値"))
    st_start_date = st.sidebar.date_input(
        "Start date", datetime.datetime.strptime("2019-1-1", "%Y-%m-%d").date()
    )
    st_end_date = st.sidebar.date_input("End date", datetime.date.today())
    st_n_limit = st.sidebar.slider("グラフで表示する銘柄の件数", 1, 100, step=1, value=3)
    st_corr_limit = st.sidebar.slider("グラフで表示する相関係数の範囲", -1.0, 1.0, (-1.0, 1.0))

    # 株価指標と個別銘柄まとめたデータフレーム
    df_merge = get_merge_data(st_index, st_p_type)
    df_stock_kashi = get_stock_col_data("貸株残高")
    df_stock_deki = get_stock_col_data("出来高")

    # 期間絞る
    df_merge = df_merge[
        (df_merge["日付"] >= str(st_start_date)) & (df_merge["日付"] <= str(st_end_date))
    ]
    df_stock_kashi = df_stock_kashi[
        (df_stock_kashi["日付"] >= str(st_start_date))
        & (df_stock_kashi["日付"] <= str(st_end_date))
    ]
    df_stock_deki = df_stock_deki[
        (df_stock_deki["日付"] >= str(st_start_date))
        & (df_stock_deki["日付"] <= str(st_end_date))
    ]

    # 貸株残高の前日との差分
    df_stock_kashi_diff = df_stock_kashi.copy()
    for c in df_stock_kashi_diff.columns[1:]:
        df_stock_kashi_diff[c] = df_stock_kashi_diff[c].diff()

    # 出来高の前日との差分
    df_stock_deki_diff = df_stock_deki.copy()
    for c in df_stock_deki_diff.columns[1:]:
        df_stock_deki_diff[c] = df_stock_deki_diff[c].diff()

    # 基準日からの増加率計算
    df_rate = df_merge.copy()
    for c in df_rate.columns[1:]:
        base = df_rate.iloc[0][c]
        df_rate[c] = (df_rate[c] - base) / base + 100

    # 株価指標と個別銘柄の相関係数
    corr = df_rate.corr()
    corr = corr[st_index].sort_values()
    corr_high = corr[
        (corr > st_corr_limit[0]) & (corr < st_corr_limit[1]) & (corr != 1)
    ][st_n_limit * -1 :].sort_values(ascending=False)

    # plot
    st.markdown("# Start dateを基準にした株価の増加率")
    title = (
        f"{st_index}の{st_p_type}と相関({st_corr_limit[0]}<r<{st_corr_limit[1]})がある銘柄のみ表示"
    )
    if len(corr_high) > 0:
        st.write(
            px.line(
                df_rate,
                x="日付",
                y=[st_index] + corr_high.index.to_list(),
                title=title,
                height=800,
                width=800,
            )
        )
    else:
        st.write(title + "なし")

    st.write("\n\n")
    st.write("### 株価の増加率の相関係数一覧")
    st.dataframe(
        pd.DataFrame(corr.sort_values(ascending=False)),
        height=500,
        width=800,
    )

    # plot
    st.write("\n")
    st.markdown("# 貸株残高の前日との差分")
    st.write(
        px.line(
            df_stock_kashi_diff,
            x="日付",
            y=corr_high.index.to_list(),
            height=800,
            width=800,
        )
    )

    # plot
    st.markdown("# 出来高の前日との差分")
    st.write(
        px.line(
            df_stock_deki_diff,
            x="日付",
            y=corr_high.index.to_list(),
            height=800,
            width=800,
        )
    )


if __name__ == "__main__":
    main()
