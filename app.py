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
def get_data(st_index, col):
    # 株価指標のデータロード
    df_index = load_df(index_dir, st_index)
    df_index = df_index[["日付", col]].rename(columns={col: st_index})

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

    # 株価指標と個別銘柄マージ
    df_merge = pd.merge(df_index, df_stocks, on="日付", how="outer")

    return df_merge


def main():
    st.markdown("# Start dateを基準にした株価の増加率")

    # サイドバー
    st_index = st.sidebar.selectbox("比較基準の株価指標", ("nikkei225", "jpx400", "topix"))
    st_p_type = st.sidebar.selectbox("価格の種類", ("始値", "終値", "高値", "安値"))

    # 株価指標と個別銘柄まとめたデータフレーム
    df_merge = get_data(st_index, st_p_type)

    # サイドバー
    st_start_date = st.sidebar.date_input(
        "Start date", datetime.datetime.strptime("2019-1-1", "%Y-%m-%d").date()
    )
    st_end_date = st.sidebar.date_input("End date", datetime.date.today())
    st_n_limit = st.sidebar.slider("表示する銘柄の件数", 1, 100, step=1, value=10)

    # 期間絞る
    df_merge = df_merge[
        (df_merge["日付"] >= str(st_start_date)) & (df_merge["日付"] <= str(st_end_date))
    ]

    # 基準日からの増加率計算
    for c in df_merge.columns[1:]:
        df_merge[c] = (df_merge[c] - df_merge.iloc[-1][c]) / df_merge.iloc[-1][c] + 100

    # 株価指標と個別銘柄の相関係数
    corr = df_merge.corr()
    corr = corr[st_index].sort_values()
    corr_high = corr[(corr > 0.7) & (corr != 1)][st_n_limit * -1 :]
    corr_low = corr[(corr < -0.7)][st_n_limit * -1 :]

    # plot
    title = f"{st_index}の{st_p_type}と高い正の相関(>0.7)がある銘柄のみ表示"
    if len(corr_high) > 0:
        st.write(
            px.line(
                df_merge,
                x="日付",
                y=[st_index] + corr_high.index.to_list(),
                title=title,
                height=800,
                width=800,
            )
        )
    else:
        st.write(title + "なし")

    title = f"{st_index}の{st_p_type}と高い負の相関(<-0.7)がある銘柄のみ表示"
    if len(corr_low) > 0:
        st.write(
            px.line(
                df_merge,
                x="日付",
                y=[st_index] + corr_low.index.to_list(),
                title=title,
                height=800,
                width=800,
            )
        )
    else:
        st.write(title + "なし")


if __name__ == "__main__":
    main()
