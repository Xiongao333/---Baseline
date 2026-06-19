import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

print("=====余额宝申赎预测 =====")
try:
    # 全局清洗函数：统一处理脏字符、空值
    def clean_numeric(series):
        series = series.astype(str).str.replace("，", "").str.replace(" ", "").str.strip()
        return pd.to_numeric(series, errors="coerce").fillna(0)

    # 1. 读取用户画像
    print("1/4 加载 user_profile_table.csv")
    df_user = pd.read_csv("user_profile_table.csv")
    df_user["constellation"] = df_user["constellation"].astype("category").cat.codes
    # 不再使用Sex列避免KeyError，跳过男女聚合

    # 2. 用户每日申赎流水
    print("2/4 加载 user_balance_table.csv")
    df_balance = pd.read_csv("user_balance_table.csv")
    df_balance["report_date"] = pd.to_datetime(df_balance["report_date"], format="%Y%m%d")
    amount_cols = [
        "tBalance", "yBalance", "total_purchase_amt", "direct_purchase_amt",
        "purchase_bal_amt", "purchase_bank_amt", "total_redeem_amt",
        "consume_amt", "transfer_amt", "tftobal_amt", "tftocard_amt", "share_amt",
        "category1", "category2", "category3", "category4"
    ]
    for c in amount_cols:
        df_balance[c] = clean_numeric(df_balance[c])

    # 3. 收益率表
    print("3/4 加载 mfd_day_share_interest.csv")
    df_yield = pd.read_csv("mfd_day_share_interest.csv")
    df_yield["mfd_date"] = pd.to_datetime(df_yield["mfd_date"], format="%Y%m%d")
    df_yield.rename(columns={"mfd_date": "report_date"}, inplace=True)
    df_yield["mfd_daily_yield"] = clean_numeric(df_yield["mfd_daily_yield"])
    df_yield["mfd_7daily_yield"] = clean_numeric(df_yield["mfd_7daily_yield"])

    # 4. Shibor利率表
    print("4/4 加载 mfd_bank_shibor.csv")
    df_shibor = pd.read_csv("mfd_bank_shibor.csv")
    df_shibor["mfd_date"] = pd.to_datetime(df_shibor["mfd_date"], format="%Y%m%d")
    df_shibor.rename(columns={"mfd_date": "report_date"}, inplace=True)
    shibor_cols = ["Interest_O_N","Interest_1_W","Interest_2_W","Interest_1_M","Interest_3_M","Interest_6_M","Interest_9_M","Interest_1_Y"]
    for c in shibor_cols:
        df_shibor[c] = clean_numeric(df_shibor[c])

    # 聚合日度资金总量
    print("\n===== 聚合全平台日度资金特征 =====")
    daily_agg = df_balance.groupby("report_date").agg(
        total_purchase=("total_purchase_amt", "sum"),
        total_redeem=("total_redeem_amt", "sum"),
        total_balance=("tBalance", "sum"),
        total_consume=("consume_amt", "sum"),
        total_share=("share_amt", "sum"),
        user_cnt=("user_id", "nunique")
    ).reset_index()

    # 合并宏观利率/收益特征
    df_macro = pd.merge(df_yield, df_shibor, on="report_date", how="outer")
    df_train_base = pd.merge(daily_agg, df_macro, on="report_date", how="left")
    df_train_base = df_train_base.sort_values("report_date").reset_index(drop=True)

    # 多周期滞后特征 1/3/7/14/21/30天
    lag_base_feats = [
        "total_balance", "total_consume", "total_share",
        "mfd_daily_yield", "mfd_7daily_yield",
        "Interest_O_N", "Interest_1_W", "Interest_2_W", "Interest_1_M",
        "Interest_3_M", "Interest_6_M", "Interest_9_M", "Interest_1_Y"
    ]
    for lag in [1,3,7,14,21,30]:
        for f in lag_base_feats:
            df_train_base[f"{f}_lag{lag}"] = clean_numeric(df_train_base[f].shift(lag))
    df_train_base = df_train_base.dropna().reset_index(drop=True)
    print(f"有效训练样本量：{len(df_train_base)}")

    # 时间周期特征：月份、星期、周末、月初月末
    df_train_base["month"] = df_train_base["report_date"].dt.month
    df_train_base["weekday"] = df_train_base["report_date"].dt.weekday
    df_train_base["is_weekend"] = (df_train_base["weekday"] >=5).astype(int)
    df_train_base["is_month_start"] = (df_train_base["report_date"].dt.day <=5).astype(int)
    df_train_base["is_month_end"] = (df_train_base["report_date"].dt.day >=26).astype(int)

    # 划分特征与标签
    feature_cols = [x for x in df_train_base.columns if x not in ["report_date","total_purchase","total_redeem"]]
    X = df_train_base[feature_cols].astype(np.float64)
    y_pur = df_train_base["total_purchase"].astype(np.float64)
    y_red = df_train_base["total_redeem"].astype(np.float64)
    # 赎回样本加权，匹配赛题55%总分权重
    weight_red = np.ones(len(y_red)) * 1.2

    # 训练LGB+XGB树模型
    print("\n===== 训练LGB+XGB融合模型 =====")
    # 申购模型
    lgb_pur = LGBMRegressor(n_estimators=400, max_depth=9, learning_rate=0.03, random_state=42, verbose=-1)
    xgb_pur = XGBRegressor(n_estimators=400, max_depth=9, learning_rate=0.03, random_state=42)
    lgb_pur.fit(X, y_pur)
    xgb_pur.fit(X, y_pur)

    # 赎回加权模型
    lgb_red = LGBMRegressor(n_estimators=450, max_depth=10, learning_rate=0.025, random_state=42, verbose=-1)
    xgb_red = XGBRegressor(n_estimators=450, max_depth=10, learning_rate=0.025, random_state=42)
    lgb_red.fit(X, y_red, sample_weight=weight_red)
    xgb_red.fit(X, y_red, sample_weight=weight_red)
    print("模型训练完成")

    # 滚动预测2014年9月30天
    print("\n===== 滚动预测9月全部日期 =====")
    pred_dates = pd.date_range(start="20140901", periods=30, freq="D")
    history_win = df_train_base.tail(30).copy()
    submit_list = []
    pur_mean = y_pur.mean()
    red_mean = y_red.mean()

    for idx, dt in enumerate(pred_dates):
        input_raw = history_win.iloc[-1][feature_cols].to_frame().T
        input_row = input_raw.astype(np.float64)

        # 双模型加权融合预测
        pred_lgb_pur = lgb_pur.predict(input_row)[0]
        pred_xgb_pur = xgb_pur.predict(input_row)[0]
        pred_pur_raw = 0.6 * pred_lgb_pur + 0.4 * pred_xgb_pur

        pred_lgb_red = lgb_red.predict(input_row)[0]
        pred_xgb_red = xgb_red.predict(input_row)[0]
        pred_red_raw = 0.6 * pred_lgb_red + 0.4 * pred_xgb_red

        # 均值修正防止极端误差
        pred_pur = max(int((pred_pur_raw * 0.85 + pur_mean * 0.15)), 10000)
        pred_red = max(int((pred_red_raw * 0.85 + red_mean * 0.15)), 10000)
        date_str = dt.strftime("%Y%m%d")
        submit_list.append([date_str, pred_pur, pred_red])

        if (idx+1) % 5 == 0:
            print(f"进度 {idx+1}/30 | {date_str} | 申购:{pred_pur} 赎回:{pred_red}")

        # 更新时序窗口
        new_line = history_win.iloc[-1].copy()
        new_line["report_date"] = dt
        for lag in [1,3,7,14,21,30]:
            for f in lag_base_feats:
                lag_col = f"{f}_lag{lag}"
                new_line[lag_col] = clean_numeric(pd.Series([new_line[f]]).shift(lag)).iloc[0]
        history_win = pd.concat([history_win, pd.DataFrame([new_line])]).iloc[1:].reset_index(drop=True)

    # 输出天池标准提交文件
    print("\n===== 生成 tc_comp_predict_table.csv =====")
    df_out = pd.DataFrame(submit_list)
    df_out.to_csv("tc_comp_predict_table.csv", index=False, header=False)

    print("\n========== 前10行预测结果 ==========")
    print("report_date,purchase,redeem")
    for line in submit_list[:10]:
        print(f"{line[0]},{line[1]},{line[2]}")
    print("文件生成完毕")

except Exception as err:
    print(f"\n!!!!!!!! 运行报错：{err} !!!!!!!!")
    import traceback
    traceback.print_exc()

input("\n按回车键关闭窗口...")
