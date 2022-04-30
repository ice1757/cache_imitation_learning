from urllib import request
import numpy as np
import copy
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def save_access_data(file_name, access_list):
    f = open('./'+file_name+'.txt', "w")
    f.write(str(access_list))
    f.close()
    print('successfully save dataset to', './'+file_name+'.txt')


df_train = None
df_valid = None

T = 25  # shot duration
M = 250  # 內容種類數量
alpha = 0.8  # 冪律參數
u_mean = 2 # 平均熱度 lambda

zm = np.random.rand(M)  # zm用來計算um，數組長度為內容種類數量，從[0,1]中取值
all_m_pop = zm**(-alpha)*u_mean*(1-alpha) # 內容m的平均熱度

requestnum = all_m_pop * T #內容m 個別總共的請求數量
wn = [] #單個內容請求到達的時間
m_start_wn = {} #每個內容的請求到達的時間
wn.append(0)
T_interval = []
for onem in range(1, M):
    T_interval = np.random.exponential(1/all_m_pop[onem],int(requestnum[onem]))
    for onetime in range(1,len(T_interval)+1):
        wn.append(onem*5+wn[onetime -1]+ T_interval[onetime - 1])
    m_start_wn[onem] = copy.copy(wn)
    wn = []
    wn.append(0)


totallist = [] #將所有內容的請求時間序列放在一起
for onekey in m_start_wn.keys():
    totallist += m_start_wn[onekey][1:]
totallist.sort() #將其按從小到達排序

num = 0
requestseq = [-1]*len(totallist)#最後生成的請求序列
for onem in range(1, M): #對每個內容做遍歷，依據其請求到達的時間節點決定請求序列中的位置
    num += 1
    wn = m_start_wn[onem]
    for onet in wn[1:]:
        requestseq[totallist.index(onet)] = onem

## for IL
hex_requestseq = [hex(i) for i in requestseq]
tmp_train = pd.DataFrame({'pc': hex(0), 'address': hex_requestseq})
tmp_valid = pd.DataFrame({'pc': hex(0), 'address': hex_requestseq[len(requestseq)//3*2:]})
if (df_train is None) and (df_valid is None):
    df_train = tmp_train
    df_valid = tmp_valid
else:
    df_train = pd.concat((df_train, tmp_train), axis = 0)
    df_valid = pd.concat((df_valid, tmp_valid), axis = 0)
if len(hex_requestseq) >= 9000 and len(hex_requestseq) <= 9999:
    df_train.to_csv(f'./dataset/req_trace/snm_{M}_train.csv', index = False, header = False)
    df_valid.to_csv(f'./dataset/req_trace/snm_{M}_valid.csv', index = False, header = False)
    save_access_data(f'snm_{M}_', requestseq)

print(len(requestseq))