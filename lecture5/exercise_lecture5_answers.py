import os
import glob
import natsort
import pandas as pd
import numpy as np
import copy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"
from sklearn import linear_model
clf = linear_model.LinearRegression()
import basic_functions as b_func
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def get_min_sleep_hr(sleep_time, df_hr, df_analysis_data):

    # 取得されたデータの睡眠時心拍の抽出
    tmp_df_sleep_hr = df_hr[df_hr['sleep_flg']==1]# df_hrを用いて睡眠中の心拍数を抽出
    tmp_df_sleep_hr.drop(index=tmp_df_sleep_hr[tmp_df_sleep_hr['hr_value']==0].index, inplace=True)

    # 1日ごとの睡眠時心拍の抽出し、1日ごｔの睡眠時心拍の最小値を導出
    tmp_dict = dict()
    for date in sleep_time:
        start = sleep_time[date]['start']
        end = sleep_time[date]['end']
        sleep_hr = tmp_df_sleep_hr.loc[start:end, 'hr_value'].values
        min_sleep_hr = np.min(sleep_hr)
        tmp_dict[date] = min_sleep_hr

    df_analysis_data['min_sleep_hr'] = [0] * len(df_analysis_data.index.tolist())
    for date in tmp_dict:
        df_analysis_data.loc[date, 'min_sleep_hr'] = tmp_dict[date]

    return df_analysis_data

#========================================
# Conducting support vector machine
#========================================
def SVM(input_data, target_data):

    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, shuffle=True, stratify=target_data)

    param_list = [0.001,0.01,0.1,1,10,100]

    best_acc = 0
    best_gamma = 0
    best_C = 0
    y_best = []

    for gamma in param_list:
        for C in param_list:
            clf = SVC(kernel='linear', gamma=gamma,C=C)
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            if test_acc > best_acc:
                best_acc = test_acc
                best_gamma = gamma
                best_C = C
                y_best = y_test_pred

    return best_acc, best_gamma, best_C, y_best, y_test

def SVM_for_estimation_of_condition(df_analysis_data, save_path):

    input_data = df_analysis_data[['min_sleep_hr']].values
    target_data = df_analysis_data['RHR'].values

    # ラベル作成
    threshold = np.median(target_data)
    target_data[target_data <= threshold] = 0
    target_data[target_data > threshold] = 1

    best_score, best_gamma, best_C, y_pred, y_test = SVM(input_data, target_data.reshape(-1, 1))
    print('best: {}, best gamma: {}, best_C: {}'.format(best_score, best_gamma, best_C))

    tp, fn, fp, tn = confusion_matrix(y_test.reshape(-1, 1), y_pred).reshape(-1)
    df_cmx = pd.DataFrame([[tp, fn], [fp, tn]], index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.8)
    sns.heatmap(df_cmx, annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    Sensitivity = tp / (tp + fn)
    Specificity = tn / (fp + tn)
    BACC = (Sensitivity + Specificity) / 2
    print('BACC: {}, Sensitivity: {}, Specificity: {}'.format(BACC, Sensitivity, Specificity))

def main():

    #==============================
    # データセット構築・解析のための準備
    #==============================
    # 使用するデータのパスなどの設定
    cwd = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    file_path = os.path.join(cwd, 'analysis_data')
    fitbit_path = glob.glob(os.path.join(file_path, 'activities', '*'))
    hr_path = os.path.join(fitbit_path[0], 'hr')
    sleep_path = os.path.join(fitbit_path[0], 'sleep')

    # データファイルのパスを取得
    hr_data_paths = natsort.natsorted(glob.glob(os.path.join(hr_path, '*.txt')))
    sleep_data_paths = natsort.natsorted(glob.glob(os.path.join(sleep_path, '*.txt')))

    # 生データの読み込み
    df_analysis_data = b_func.get_RHR(hr_data_paths=hr_data_paths)
    df_hr, sleep_time = b_func.get_sleep_hr_data(hr_data_paths, sleep_data_paths)
    df_hr['time'] = pd.to_datetime(df_hr['time'])
    df_hr = df_hr.reset_index(drop=True)
    df_hr = df_hr.set_index('time')
    df_analysis_data = df_analysis_data.set_index('Date')

    #==============================
    # データセット構築
    #==============================
    df_analysis_data = get_min_sleep_hr(sleep_time, df_hr, df_analysis_data)

    save_path = os.path.join(cwd, 'analysis_results', 'lecture5')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # SVMの実行
    SVM_for_estimation_of_condition(df_analysis_data, save_path)

if __name__ == '__main__':
    main()