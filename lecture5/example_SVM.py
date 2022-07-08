import os
import pandas as pd
import numpy as np
import copy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import seaborn as sns

#========================================
# Conducting support vector machine
#========================================
def SVM(input_data, target_data):

    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, shuffle=True, random_state=0, stratify=target_data)

    param_list = [0.001,0.01,0.1,1,10,100]

    best_acc = 0
    best_gamma = 0
    best_C = 0
    y_best = []

    for gamma in param_list:
        for C in param_list:
            clf = SVC(kernel='linear', gamma=gamma, C=C)
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            if test_acc > best_acc:
                best_acc = test_acc
                best_gamma = gamma
                best_C = C
                y_best = y_test_pred

    return best_acc, best_gamma, best_C, y_best, y_test

def SVM_for_vehicle_data(dataset, save_path):

    input_data = dataset[['engine-size']].values
    tmp_target_data = dataset['price'].values

    # priceの大、小の分類
    target_data = copy.deepcopy(tmp_target_data)
    target_data[target_data < np.median(tmp_target_data)] = 0
    target_data[target_data >= np.median(tmp_target_data)] = 1

    best_score, best_gamma, best_C, y_pred, y_test = SVM(input_data, target_data.reshape(-1, 1))
    print('best: {}, best gamma: {}, best_C: {}'.format(best_score, best_gamma, best_C))

    tp, fn, fp, tn = confusion_matrix(y_test.reshape(-1, 1), y_pred).reshape(-1)
    df_cmx = pd.DataFrame([[tp, fn], [fp, tn]], index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.8)
    sns.heatmap(df_cmx, annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_confusion_matrix.png'))
    Sensitivity = tp / (tp + fn)
    Specificity = tn / (fp + tn)
    BACC = (Sensitivity + Specificity) / 2
    print('BACC: {}, Sensitivity: {}, Specificity: {}'.format(BACC, Sensitivity, Specificity))

def main():

    # 使用するデータのパスなどの設定
    cwd = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cwd, 'data', 'strong_cor.csv')
    save_path = os.path.join(cwd, 'results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df = pd.read_csv(file_path)

    # SVMの実行
    SVM_for_vehicle_data(df, save_path)

if __name__ == '__main__':
    main()