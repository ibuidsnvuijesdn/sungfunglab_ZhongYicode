# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:38:41 2025

@author: User
"""

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess


# 保留你提供的這部分
repository_input1 = r'D:\Desktop\碩士檔\質譜分析實驗室\實驗資料\Metabolites_data\MSDIAL\excel\msp_match_5_1\20231222_SP_Amide_10mMAmF_0.125_FA_POS_SWATH_oldlib\dot_product_450_0_630'
relative_path1 = r'ALL_match_Feature_MS_DIAL_Ref_Sug_cpn_dot_product_450_0_630.xlsx'
file_path1 = os.path.join(repository_input1, relative_path1)
sheet_name1 = 'Fn_ID_RT_MZ_CPN_Int'
Filename_PeakID_RT_MZ_Intensity = pd.read_excel(file_path1, sheet_name=sheet_name1, header=None)

Filename = pd.concat([Filename_PeakID_RT_MZ_Intensity.iloc[[0],0:4],Filename_PeakID_RT_MZ_Intensity.iloc[[0],26:41]], ignore_index=True, axis=1)
Filename_QC_SP = pd.concat([Filename_PeakID_RT_MZ_Intensity.iloc[[0],0:4],Filename_PeakID_RT_MZ_Intensity.iloc[[0],23:41]], ignore_index=True, axis=1)
Filename_QC_SP = pd.concat([Filename_PeakID_RT_MZ_Intensity.iloc[[0],0:4],Filename_PeakID_RT_MZ_Intensity.iloc[[0],23:41]], ignore_index=True, axis=1)
Filename_BK_QC_SP = pd.concat([Filename_PeakID_RT_MZ_Intensity.iloc[[0],0:4],Filename_PeakID_RT_MZ_Intensity.iloc[[0],4:41]], ignore_index=True, axis=1)
Filename_BK_QC_SP = pd.concat([Filename_PeakID_RT_MZ_Intensity.iloc[[0],0:4],Filename_PeakID_RT_MZ_Intensity.iloc[[0],4:41]], ignore_index=True, axis=1)
PeakID_MZ_RT = pd.concat([Filename_PeakID_RT_MZ_Intensity.iloc[1:,0:4]], ignore_index=True)
#如果要將peak area取log就寫1；如果不要將peak area取log就寫0
log_peak_area = 1
if log_peak_area == 1:
    BK_array = np.log10(np.array(Filename_PeakID_RT_MZ_Intensity.iloc[1:,4:23].astype(float)) + 1*10**0)
    QC_array = np.log10(np.array(Filename_PeakID_RT_MZ_Intensity.iloc[1:,23:26].astype(float)) + 1*10**0)
    SP_array = np.log10(np.array(Filename_PeakID_RT_MZ_Intensity.iloc[1:,26:41].astype(float)) + 1*10**0)
    log_peak_area_filename = "log_"
if log_peak_area == 0:
    BK_array = np.array(Filename_PeakID_RT_MZ_Intensity.iloc[1:,4:23].astype(float))
    QC_array = np.array(Filename_PeakID_RT_MZ_Intensity.iloc[1:,23:26].astype(float))
    SP_array = np.array(Filename_PeakID_RT_MZ_Intensity.iloc[1:,26:41].astype(float))
    log_peak_area_filename = ""

# --- index 設定 ---
# 定義 QC 與 SP 的欄位索引
bk_idx = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36])
qc_idx = np.array([1, 13, 25])
sp_idx = np.array([3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 27, 29, 31, 33, 35])
qc_sp_idx = np.sort(np.concatenate([qc_idx, sp_idx]))
bk_positions = bk_idx + 1
qc_positions = qc_idx + 1
sp_positions = sp_idx + 1
qc_sp_positions = qc_sp_idx + 1

# --- 合併 QC + SP 原始資料 ---
QC_SP_array = np.concatenate([QC_array, SP_array], axis=1)

# --- Batch effect correction ---
# (1) BK regression 預測 QC+SP baseline
x_bk = bk_positions.reshape(-1, 1)
x_qcsp = qc_sp_positions.reshape(-1, 1)
corrected = np.empty_like(QC_SP_array)

for i in range(BK_array.shape[0]):
    model = LinearRegression()
    model.fit(x_bk, BK_array[i, :])
    baseline_pred = model.predict(x_qcsp)
    corrected[i, :] = QC_SP_array[i, :] - baseline_pred

corrected = np.maximum(corrected, 0)


# (3) LOWESS trend fitting on QC samples → 插值應用於 QC+SP
# 加權多項式回歸擬合
def weighted_polyfit(x, y, degree=2, weights=None):
    if weights is None:
        weights = np.ones_like(x)
    X = np.vander(x, N=degree + 1, increasing=True)
    W = np.diag(weights)
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    return beta

# 多項式預測
def weighted_polyval(beta, x):
    X = np.vander(x, N=len(beta), increasing=True)
    return X @ beta

lowess_corrected = np.empty_like(corrected)
for i in range(corrected.shape[0]):
    # 假設第 i 個 feature
    x_qc = qc_positions.flatten()           # QC 位置 (1D)
    y_qc = QC_array[i, :]                   # 對應強度 (1D)
    x_all = qc_sp_positions                 # QC + SP 全部位置
    # 設定權重（可以設成全 1，或用 tricubic kernel）
    weights = np.ones_like(x_qc)            # 簡單先用等權重
    # 擬合二次多項式
    beta = weighted_polyfit(x_qc, y_qc, degree=2, weights=weights)
    # 預測所有 QC+SP 的趨勢值
    interp = weighted_polyval(beta, x_all)
    epsilon = 1e-1
    interp[interp < epsilon] = epsilon
    lowess_corrected[i, :] = corrected[i, :] / interp

# (4) 乘回第一個 QC 值
qc_first = QC_array[:, 0].reshape(-1, 1)
final_corrected_qc_sp_array = lowess_corrected * qc_first

# --- 分離校正後的 QC 與 SP ---
corrected_QC_array = final_corrected_qc_sp_array[:, np.isin(qc_sp_idx, qc_idx)]
corrected_SP_array = final_corrected_qc_sp_array[:, np.isin(qc_sp_idx, sp_idx)]

PeakID_RT_MZ_Batch_effect_internal = pd.concat([PeakID_MZ_RT, pd.DataFrame(corrected_SP_array)] ,axis=1 ,ignore_index=True)
Filename_PeakID_RT_MZ_Batch_effect_internal = pd.concat([Filename, pd.DataFrame(PeakID_RT_MZ_Batch_effect_internal)] ,axis=0 ,ignore_index=True)
PeakID_RT_MZ_Batch_effect_internal_QC_SP = pd.concat([PeakID_MZ_RT, pd.DataFrame(corrected_QC_array), pd.DataFrame(corrected_SP_array)] ,axis=1 ,ignore_index=True)
Filename_PeakID_RT_MZ_Batch_effect_internal_QC_SP = pd.concat([Filename_QC_SP, pd.DataFrame(PeakID_RT_MZ_Batch_effect_internal_QC_SP)] ,axis=0 ,ignore_index=True)

def concentration_transform(A):
    # 假設你已有 df
    df = A.copy()
    
    # 找出包含 "Isoleucine" 的 row（在第3欄，也就是 column index 3）
    target_row = df[df.iloc[:, 3].astype(str).str.contains("Isoleucine", case=False)]
    
    # 如果找到 Isoleucine row
    if not target_row.empty:
        # 取第一個找到的 row 的欄位 4~21 當除數
        divisor = target_row.iloc[0, 4:].astype(float).values
    
        # 防止除以0（可選）
        divisor[divisor == 0] = 1e-6
    
        # 對原始 df 的第1列之後，第4~21欄做除法再乘50
        df.iloc[1:, 4:] = df.iloc[1:, 4:].astype(float).div(divisor).multiply(50)
    
        # 若要儲存結果，可用：
        # df.to_excel("scaled_result.xlsx", index=False)
    
    
    else:
        print("❌ 找不到包含 'Isoleucine' 的列")

    return df

Filename_PeakID_RT_MZ_Batch_effect_internal_scaled = concentration_transform(Filename_PeakID_RT_MZ_Batch_effect_internal)
Filename_PeakID_RT_MZ_Batch_effect_internal_QC_SP_scaled = concentration_transform(Filename_PeakID_RT_MZ_Batch_effect_internal_QC_SP)

repository_output = r'D:\Desktop\碩士檔\質譜分析實驗室\實驗資料\Metabolites_data\MSDIAL\excel\msp_match_5_1\20231222_SP_Amide_10mMAmF_0.125_FA_POS_SWATH_oldlib\dot_product_430_0_600_compound_name'
Relative_path = f'ALL_match_Feature_MD_RS_bei_cpn_ISconc_{log_peak_area_filename}trend_bkqc_wpr_dot_product_450_0_630.xlsx'
result_path = os.path.join(repository_output, Relative_path)
with pd.ExcelWriter(result_path) as writer:
    pd.DataFrame(corrected_SP_array).to_excel(writer, sheet_name='Batch_effect_internal', index=False, header=False)
    pd.DataFrame(Filename_PeakID_RT_MZ_Batch_effect_internal).to_excel(writer, sheet_name='Fn_ID_RT_MZ_Beff_internal', index=False, header=False)
    pd.DataFrame(final_corrected_qc_sp_array).to_excel(writer, sheet_name='Batch_effect_internal_QC_SP', index=False, header=False)
    pd.DataFrame(Filename_PeakID_RT_MZ_Batch_effect_internal_QC_SP).to_excel(writer, sheet_name='Fn_ID_RT_MZ_Beff_internal_QC_SP', index=False, header=False)
    pd.DataFrame(Filename_PeakID_RT_MZ_Batch_effect_internal_scaled).to_excel(writer, sheet_name='Fn_ID_RT_MZ_Bei_sca', index=False, header=False)
    pd.DataFrame(Filename_PeakID_RT_MZ_Batch_effect_internal_QC_SP_scaled).to_excel(writer, sheet_name='Fn_ID_RT_MZ_Bei_QC_SP_sca', index=False, header=False)