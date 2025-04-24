# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 21:12:39 2025

@author: Yi-Chun Shih
"""
#%% packages for all analysis
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import kstest
from scipy.stats import mannwhitneyu
from scipy.stats import sem
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from clampsuite import ExpManager
from fi_plot import multiline_plot
from statsandplots.plotting import CategoricalPlot
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


#%% mEPSC data analysis

def get_data(df, directory, columns, group, unique_id, sheet="Summary"):
    paths = list(Path(directory).rglob("*.xlsx"))
    df_dict = {}
    for i in paths:
        key = f"{i.parts[-3]}_{i.stem}"
        df_dict[key] = pd.read_excel(i, sheet_name=sheet)

    data_dict = {}
    for i in range(df.shape[0]):
        if df[group].iloc[i] in data_dict:
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]
        else:
            data_dict[df[group].iloc[i]] = {}
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]

    all_dict = {}
    for k in columns:
        temp_dict = {}
        all_dict[k] = temp_dict
        for i in data_dict.keys():
            temp_dict[i] = {}
            for j in data_dict[i].keys():
                temp_dict[i][j] = data_dict[i][j][k]
            temp_dict[i] = pd.DataFrame(temp_dict[i])
    return all_dict

def spline_cdf(data, size=500):
    data_sort = np.sort(data)
    data_sortx = np.arange(data_sort.size) / data_sort.size
    cs = interpolate.CubicSpline(data_sortx, data_sort, bc_type="clamped")
    cdf_y = np.arange(0, size) / size
    cdf_x = cs(cdf_y)
    return(cdf_x)

def get_averaged_cdf_by_spline(data, column, group, cdf_size):
    size = data[column][group].shape[1]
    df_cdf = pd.DataFrame(index=range(cdf_size), columns=range(size))
    for i in range(size):
        col_values = data[column][group].iloc[:, i].dropna()
        x1 = spline_cdf(col_values, cdf_size)
        df_cdf.iloc[:, i] = x1
    cdf_mean = df_cdf.mean(axis=1)
    return(cdf_mean)

def get_cell_cdf_by_spline(data, column, group, cdf_size):
    size = data[column][group].shape[1]
    df_cdf = pd.DataFrame(index=range(cdf_size), columns=range(size))
    for i in range(size):
        col_values = data[column][group].iloc[:, i].dropna()
        x1 = spline_cdf(col_values, cdf_size)
        df_cdf.iloc[:, i] = x1
    return(df_cdf)

def string_pvalue(pvalue):
    if pvalue >= 0.01:
        p_print = f"{round(pvalue, 3)}"
    elif pvalue >= 0.001:
        p_print = f"{round(pvalue, 3)}"
    else:
        p_print = f"{pvalue:.2e}"
    return(p_print)

def plot_cdf_with_error_band (data_dict, column, cdf_size, ax, xlim_right=None):
    df1 = get_cell_cdf_by_spline(data_dict,column,"WT",cdf_size).transpose()
    df2 = get_cell_cdf_by_spline(data_dict,column,"KO",cdf_size).transpose()
    df1.insert(0,"Genotype","WT")
    df2.insert(0,"Genotype","KO")
    df = pd.concat([df1,df2], axis=0)
    plt_y = np.arange(1, cdf_size+1) / cdf_size
    
    wt_values = df[df['Genotype'] == 'WT'].drop(columns='Genotype')
    ko_values = df[df['Genotype'] == 'KO'].drop(columns='Genotype')
    wt_values = wt_values.apply(pd.to_numeric, errors='coerce').values
    ko_values = ko_values.apply(pd.to_numeric, errors='coerce').values
    mean_WT = np.mean(wt_values, axis=0)
    sem_WT = np.std(wt_values, axis=0, ddof=1) / np.sqrt(wt_values.shape[0])
    mean_KO = np.mean(ko_values, axis=0)
    sem_KO = np.std(ko_values, axis=0, ddof=1) / np.sqrt(ko_values.shape[0])
    
    ax.plot(mean_WT, plt_y, label='WT', color='black')
    ax.fill_betweenx(plt_y, mean_WT - sem_WT, mean_WT + sem_WT, color='black', alpha=0.2)
    ax.plot(mean_KO, plt_y, label='KO', color='red')
    ax.fill_betweenx(plt_y, mean_KO - sem_KO, mean_KO + sem_KO, color='red', alpha=0.2)
    ax.set_xlabel(column, fontsize=25)
    ax.set_ylabel("Cumulative fraction", fontsize=25)
    ax.set_ylim(0,1)
    if xlim_right is not None:
        current_xlim = ax.get_xlim()
        ax.set_xlim(left=current_xlim[0], right=xlim_right)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=20, bottom=True, left=True, width=2)
    
    return ax

# path of data
path_sum = "path of excel file"
path_data = "path of data folder"

# import data for cdf
df = pd.read_excel(path_sum) # directory of summery excel file
# the excel file name need to be exactly the same as unique id, so check filename use 'cell' or 'epoch'
df["id"] = df["Date"] + "_" + "cell" + "_" + df["Epoch"].astype(str) + "_" + df["Recorder"].astype(str) 
df.drop(df[df["Included"]=="No"].index, inplace=True)
df.drop(df[df["Recording"]=="mIPSC"].index, inplace=True) # drop mIPSC
df = df.sort_values(by=['Genotype'], ascending=False)
wt_count = df[df["Genotype"] == "WT"].shape[0]
ko_count = df[df["Genotype"] == "KO"].shape[0]
print(f"{wt_count} WT cells and {ko_count} KO cells")
directory = path_data # directory of data folder
columns = [
    "Amplitude (pA)",
    "IEI (ms)",
    "Est tau (ms)",
    "Rise time (ms)",
    "Rise rate (pA/ms)",
]
group = "Genotype"
unique_id = "id"
sheet = "Raw data"

# extract data from sheet "Raw data"
all_dict = get_data(df, directory, columns, group, unique_id, sheet)

# covert iei into frequency
frequency_dict = {}
for genotype in ['WT', 'KO']:
    ieims_data = all_dict['IEI (ms)'][genotype]
    frequency_data = 1000 / ieims_data
    frequency_dict[genotype] = frequency_data
all_dict['Frequency (Hz)'] = frequency_dict

# KS test for CDF
for j in columns:
    cdf_mean_wt = get_averaged_cdf_by_spline(all_dict,j,"WT",500)
    cdf_mean_ko = get_averaged_cdf_by_spline(all_dict,j,"KO",500)
    pvalue = kstest(cdf_mean_wt,cdf_mean_ko)[1]
    print(f"{j}: {string_pvalue(pvalue)}")

# import data for mean
df2 = pd.read_excel(path_sum)
df2["id"] = df2["Date"] + "_" + "cell" + "_" + df2["Epoch"].astype(str) + "_" + df2["Recorder"].astype(str)
df2.drop(df2[df2["Included"]=="No"].index, inplace=True)
df2.drop(df2[df2["Recording"]=="mIPSC"].index, inplace=True) # drop mIPSC
wt_count = df[df["Genotype"] == "WT"].shape[0]
ko_count = df[df["Genotype"] == "KO"].shape[0]
print(f"{wt_count} WT cells and {ko_count} KO cells")
directory = path_data
columns = [
    "Amplitude (pA)",
    "IEI (ms)",
    "Est tau (ms)",
    "Rise time (ms)",
    "Rise rate (pA/ms)"
]
group = "Genotype"
unique_id = "id"
sheet = "Final data"

# extract data from sheet "Final data"
all_dict_final = get_data(df2, directory, columns, group, unique_id, sheet)

# covert iei into frequency
frequency_dict = {}
for genotype in ['WT', 'KO']:
    ieims_data = all_dict_final['IEI (ms)'][genotype]
    frequency_data = 1000 / ieims_data
    frequency_dict[genotype] = frequency_data
all_dict_final['Frequency (Hz)'] = frequency_dict

columns = [
    "Amplitude (pA)",
    "Frequency (Hz)",
    "Est tau (ms)",
    "Rise time (ms)",
    "Rise rate (pA/ms)",
]

df_temp = df2[["Date",'Genotype','Age','Sex','Epoch','Recording','Rin','Cm','Rs_B','Rs_A','Spines','Recorder','id']]
for i, j in enumerate(columns):
    average = pd.concat([all_dict_final[j]["WT"].iloc[0], all_dict_final[j]["KO"].iloc[0]]).reset_index().rename(columns={"index": "id", 0: j})
    if i == 0:
        df_accepted = df_temp.merge(average, on="id")
    else:
        df_accepted = df_accepted.merge(average, on="id")
df_accepted = df_accepted.sort_values(by=['Genotype'], ascending=False)
df_accepted = df_accepted.rename(columns={"Rin": "Resistance (MΩ)", "Cm": "Capacitance (pF)"})

# MW test for cell values
for j in columns:
    value_wt = df_accepted[df_accepted['Genotype'] == 'WT'][j]
    value_ko = df_accepted[df_accepted['Genotype'] == 'KO'][j]
    pvalue = mannwhitneyu(value_wt,value_ko)[1]
    print(f"{j}: {string_pvalue(pvalue)}")

# mean and std
columns = [
    "Amplitude (pA)",
    "Frequency (Hz)",
    "Est tau (ms)",
    "Resistance (MΩ)",
    "Capacitance (pF)",
    "Rise time (ms)",
    "Rise rate (pA/ms)",
]
for i in ["WT","KO"]:
    for _, j in enumerate(columns):
        col_mean = df_accepted[df_accepted['Genotype'] == i][j].mean()
        col_mean = round(col_mean,1)
        col_std = df_accepted[df_accepted['Genotype'] == i][j].std()
        col_std = round(col_std,1)
        print(f"{i}_{j}: mean = {col_mean} std = {col_std}")

# test distribution by qqplot
stats.probplot(df_accepted[df_accepted['Genotype'] == 'WT']["Frequency (Hz)"], dist="norm", plot=plt)
stats.probplot(df_accepted[df_accepted['Genotype'] == 'KO']["Frequency (Hz)"], dist="norm", plot=plt)


#%% Current clamp analysis

def get_data(df, directory, columns, group, unique_id, sheet="Summary"):
    paths = list(Path(directory).rglob("*.xlsx"))
    df_dict = {}
    for i in paths:
        key = f"{i.parts[-3]}_{i.stem}"
        df_dict[key] = pd.read_excel(i, sheet_name=sheet)

    data_dict = {}
    for i in range(df.shape[0]):
        if df[group].iloc[i] in data_dict:
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]
        else:
            data_dict[df[group].iloc[i]] = {}
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]

    all_dict = {}
    for k in columns:
        temp_dict = {}
        all_dict[k] = temp_dict
        for i in data_dict.keys():
            temp_dict[i] = {}
            for j in data_dict[i].keys():
                temp_dict[i][j] = data_dict[i][j][k]
            temp_dict[i] = pd.DataFrame(temp_dict[i])
    return all_dict

def string_pvalue(pvalue):
    if pvalue >= 0.01:
        p_print = f"{round(pvalue, 3)}"
    elif pvalue >= 0.001:
        p_print = f"{round(pvalue, 3)}"
    else:
        p_print = f"{pvalue:.2e}"
    return(p_print)

# set path
path_sum = "path of excel file"
path_data = "path of data folder"

df = pd.read_excel(path_sum)
df["id"] = df["Date"] + "_" + "epoch" + "_" + df["Epoch"].astype(str) + "_" + df["Recorder"].astype(str)
df.drop(df[df["Included"]=="No"].index, inplace=True)
wt_count = df[df["Genotype"] == "WT"].shape[0]
ko_count = df[df["Genotype"] == "KO"].shape[0]
print(f"{wt_count} WT cells and {ko_count} KO cells")
directory = path_data
columns = [
    "Rheobase (pA)",
    "Delta_v (mV)",
    "Spike_threshold (mV)",
    "Spike_threshold_time (ms)",
    "Spike_peak_volt",
    "Spike_time (ms)",
    "IEI",
    "Spike_width (ms)",
    "Max_AP_vel",
    "Peak_AHP (mV)",
    "Peak_AHP (ms)",
    "Membrane resistance",
]
group = "Genotype"
unique_id = "id"
sheet = "Final data (pulse)"

all_dict_final = get_data(df, directory, columns, group, unique_id, sheet)

# measure spike rise time & spike apmlitude 
risetime_dict = {}
# for genotype in ['WT']:
for genotype in ['WT', 'KO']:
    threshold_time_data = all_dict_final['Spike_threshold_time (ms)'][genotype]
    spike_time_data = all_dict_final['Spike_time (ms)'][genotype]
    risetime_data = spike_time_data - threshold_time_data
    risetime_dict[genotype] = risetime_data
all_dict_final['Rise time (ms)'] = risetime_dict

amplitude_dict = {}
# for genotype in ['WT']:
for genotype in ['WT', 'KO']:
    threshold_data = all_dict_final['Spike_threshold (mV)'][genotype]
    spike_volt_data = all_dict_final['Spike_peak_volt'][genotype]
    amplitude_data = spike_volt_data - threshold_data
    amplitude_dict[genotype] = amplitude_data
all_dict_final['Spike amplitude (mV)'] = amplitude_dict

columns = [
    "Rheobase (pA)",
    "Spike_threshold (mV)",
    "Spike amplitude (mV)",
    "Rise time (ms)",
    "IEI",
    "Spike_width (ms)",
    "Max_AP_vel",
    "Peak_AHP (mV)",
    "Peak_AHP (ms)",
    "Membrane resistance",
]
df_temp = df[["Date",'Genotype','Age','Sex','Epoch','Cm','RMP','RMP2','Rs_B','Rs_A','Recorder','id']] #Do Not drop nan!
# df_temp = df[["Date",'Genotype','Age','Sex','Epoch','E2.tdT','Cm','RMP','RMP2','Rs_B','Rs_A','Recorder','id']]
for i, j in enumerate(columns):
    # values = all_dict_final[j]["WT"].iloc[0].reset_index().rename(columns={"index": "id", 0: j})
    values = pd.concat([all_dict_final[j]["WT"].iloc[0], all_dict_final[j]["KO"].iloc[0]]).reset_index().rename(columns={"index": "id", 0: j})
    if i == 0:
        df_accepted = df_temp.merge(values, on="id")
    else:
        df_accepted = df_accepted.merge(values, on="id")
df_accepted = df_accepted.sort_values(by=['Genotype'], ascending=False)
df_accepted = df_accepted.rename(columns={"Membrane resistance": "Rm (MΩ)", "RMP": "RMP (mV)", "Spike_threshold (mV)": "Threshold (mV)", "IEI": "ISI (ms)", "Spike_width (ms)": "Spike width (ms)", "Max_AP_vel": "Max AP velocity (mV/ms)", "Peak_AHP (mV)": "AHP amplitude (mV)", "Peak_AHP (ms)": "AHP duration (ms)"})

columns = [
    "Rm (MΩ)",
    "RMP (mV)",
    "Threshold (mV)",
    "Rheobase (pA)",
    "Spike amplitude (mV)",
    "Spike width (ms)",
    "ISI (ms)",
    # "Rise time (ms)",
    "Max AP velocity (mV/ms)",
    "AHP amplitude (mV)",
    "AHP duration (ms)"
]

# mean and std
for i in ["WT","KO"]:
    for _, j in enumerate(columns):
        col_mean = df_accepted[df_accepted['Genotype'] == i][j].mean()
        col_mean = round(col_mean,1)
        col_std = df_accepted[df_accepted['Genotype'] == i][j].std()
        col_std = round(col_std,1)
        print(f"{i}_{j}: mean = {col_mean} std = {col_std}")

# MW test
for j in columns:
    value_wt = df_accepted[df_accepted['Genotype'] == 'WT'][j].dropna()
    value_ko = df_accepted[df_accepted['Genotype'] == 'KO'][j].dropna()
    pvalue = mannwhitneyu(value_wt,value_ko)[1]
    print(f"{j}: {string_pvalue(pvalue)}")

# I-F
def get_data2(df, directory, group, unique_id, sheet="Hertz"):
    paths = list(Path(directory).rglob("*.xlsx"))
    df_dict = {}
    for i in paths:
        try:
            key = f"{i.parts[-3]}_{i.stem}"
            df_dict[key] = pd.read_excel(i, sheet_name=sheet)
        except ValueError as e:
            print(f"Skipping {i} because the {sheet} was not found.")

    data_dict = {}
    for i in range(df.shape[0]):
        if df[group].iloc[i] in data_dict:
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]
        else:
            data_dict[df[group].iloc[i]] = {}
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]

    all_dict = {}
    temp_dict = {}
    all_dict["Pulse_amp (pA)"] = temp_dict
    for i in data_dict.keys():
        temp_dict[i] = {}
        for j in data_dict[i].keys():
            temp_dict[i][j] = data_dict[i][j]["Pulse_amp (pA)"]
        temp_dict[i] = pd.DataFrame(temp_dict[i])
    
    temp_dict = {}
    all_dict["Number of spike"] = temp_dict
    for i in data_dict.keys():
        temp_dict[i] = {}
        for j in data_dict[i].keys():
            temp_dict[i][j] = data_dict[i][j].iloc[:,1]
        temp_dict[i] = pd.DataFrame(temp_dict[i])
    

    return all_dict

def number_of_spk(data, current_dict, current_drop):
    nspk_wt = data["Number of spike"]["WT"].iloc[:,:].T
    nspk_ko = data["Number of spike"]["KO"].iloc[:,:].T
    nspk_wt["Genotype"] = ["WT"]*len(nspk_wt)
    nspk_ko["Genotype"] = ["KO"]*len(nspk_ko)
    nspk = pd.concat([nspk_wt,nspk_ko], axis=0)
    nspk = nspk.rename(columns=current_dict)
    nap = nspk.drop(current_drop, axis=1)
    result = nap.melt('Genotype', var_name = 'Current', value_name = 'Number_of_AP')
    
    return (result)
#
sheet = "Hertz"
all_dict_if = get_data2(df, directory, group, unique_id, sheet)

# choose cell type
"""P15 PYR"""
current_columns={ 
    0:"-100",1:"-75",2:"-50",3:"-25",4:"0",5:"25",
    6:"50",7:"75",8:"100",9:"125",10:"150",
    11:"175",12:"200",13:"225",14:"250",15:"275",
    16:"300", 17:"325",18:"350" 
    }
current_hide=["-100","-75","-50","-25","0","25","325","350"] 

"""P15 E2"""
# current_columns={
#     0:"-100",1:"-75",2:"-50",3:"-25",4:"0",5:"25",
#     6:"50",7:"75",8:"100",9:"125",10:"150",
#     11:"175",12:"200",13:"225",14:"250",15:"275",
#     16:"300",17:"325",18:"350",19:"375",20:"400",
#     21:"425"
#     }
# current_hide=["-100","-75","-50","-25","0","25","375","400","425"] 

"""Adult PYR"""
# current_columns={
#     0:"-100",1:"-75",2:"-50",3:"-25",4:"0",5:"25",
#     6:"50",7:"75",8:"100",9:"125",10:"150",
#     11:"175",12:"200",13:"225",14:"250",15:"275",
#     16:"300"
#     }
# current_hide=["-100","-75","-50","-25","0","25"] 

"""Adult E2"""
# current_columns={
#     0:"-100",1:"-75",2:"-50",3:"-25",4:"0",5:"25",
#     6:"50",7:"75",8:"100",9:"125",10:"150",
#     11:"175",12:"200",13:"225",14:"250",15:"275",
#     16:"300",17:"325",18:"350",19:"375",20:"400",
#     21:"450"
#     }
# current_hide=["-100","-75","-50","-25","0","25","375","400","450"] 

nap_melt  = number_of_spk(all_dict_if, current_columns, current_hide)

# Two-way ANOVA for IF
model = ols('Number_of_AP~Genotype+Current+Genotype:Current',nap_melt).fit()
sm.stats.anova_lm(model, typ=2)

# I-V
def get_data3(df, directory, group, unique_id, sheet="IV lines"):
    paths = list(Path(directory).rglob("*.xlsx"))
    df_dict = {}
    for i in paths:
        key = f"{i.parts[-3]}_{i.stem}"
        df_dict[key] = pd.read_excel(i, sheet_name=sheet)

    data_dict = {}
    for i in range(df.shape[0]):
        if df[group].iloc[i] in data_dict:
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]
        else:
            data_dict[df[group].iloc[i]] = {}
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]
    all_dict = {}
    temp_dict = {}
    temp_dict = {}
    all_dict["IV"] = temp_dict
    for i in data_dict.keys():
        temp_dict[i] = {}
        for j in data_dict[i].keys():
            temp_dict[i][j] = data_dict[i][j].iloc[:,0]
        temp_dict[i] = pd.DataFrame(temp_dict[i])
    
    return all_dict

sheet = "IV lines"
all_dict_iv = get_data3(df, directory, group, unique_id, sheet)

iv_wt = all_dict_iv["IV"]["WT"].iloc[0:7,:].T
iv_ko = all_dict_iv["IV"]["KO"].iloc[0:7,:].T
iv_wt["Genotype"] = ["WT"]*len(iv_wt)
iv_ko["Genotype"] = ["KO"]*len(iv_ko)
df_iv = pd.concat([iv_wt,iv_ko], axis=0)
df_iv = df_iv.rename(columns={0:"-100",1:"-75",2:"-50",3:"-25",4:"0",5:"25",6:"50"})
iv_melt = df_iv.melt('Genotype', var_name = 'Current', value_name = 'Voltage')

# Two-way ANOVA for IV
model = ols('Voltage~Genotype+Current+Genotype:Current',iv_melt).fit()
sm.stats.anova_lm(model, typ=2)


#%% oEPSC/oIPSC analysis

def get_data(df, directory, columns, group, unique_id, rec_id, sheet):
    """load all excel files in the directory, and create a dict with all loaded data"""
    paths = list(Path(directory).rglob("*.xlsx"))
    df_dict = {}
    for i in paths:
        key = f"{i.parts[-3]}_{i.stem[-4:-3]}_{i.stem[-2:]}"
        df_dict[key] = pd.read_excel(i, sheet_name=sheet)
    
    a = df.loc[df["Included"]=="No"]["id2"].reset_index(drop=True)
    b = df.loc[df["Included"]=="No"]["Epoch"].reset_index(drop=True)
    
    for i in range(a.size):
        temp = df_dict[a[i]]
        df_dict[a[i]] = temp[temp['Epoch'] != b[i]]
    
    """Separate data based on group"""
    data_dict = {}
    for i in range(df.shape[0]):
        if df[group].iloc[i] in data_dict:
            if df[rec_id].iloc[i] in df_dict:
                temp = df_dict[df[rec_id].iloc[i]]
                data_dict[df[group].iloc[i]]["AMPA"][df[rec_id].iloc[i]] = temp.loc[
                    temp["Peak direction"]=="negative"
                    ]
                data_dict[df[group].iloc[i]]["GABA"][df[rec_id].iloc[i]] = temp.loc[
                    temp["Peak direction"]=="positive"
                    ]
        else:
            data_dict[df[group].iloc[i]] = {}
            data_dict[df[group].iloc[i]]["AMPA"] = {}
            data_dict[df[group].iloc[i]]["GABA"] = {}
            if df[rec_id].iloc[i] in df_dict:
                temp = df_dict[df[rec_id].iloc[i]]
                data_dict[df[group].iloc[i]]["AMPA"][df[rec_id].iloc[i]] = temp.loc[
                    temp["Peak direction"]=="negative"
                    ]
                data_dict[df[group].iloc[i]]["GABA"][df[rec_id].iloc[i]] = temp.loc[
                    temp["Peak direction"]=="positive"
                    ]
                
    """"""
    all_dict = {}
    for k in columns:
        temp_dict = {}
        all_dict[k] = temp_dict
        for i in data_dict.keys():
            temp_dict[i] = {} 
            for j in data_dict[i].keys():
                temp_dict[i][j] = {}
                for l in data_dict[i][j].keys():
                    temp_dict[i][j][l] = data_dict[i][j][l][k].reset_index(drop=True)
                temp_dict[i][j] = pd.DataFrame(temp_dict[i][j])
    return all_dict

def string_pvalue(pvalue):
    if pvalue >= 0.01:
        p_print = f"{round(pvalue, 2)}"
    elif pvalue >= 0.001:
        p_print = f"{round(pvalue, 3)}"
    else:
        p_print = f"{pvalue:.2e}"
    return(p_print)

path_sum = "path of excel file"
path_data = "path of data folder"

df = pd.read_excel(path_sum) # directory of summery excel file
df = df.loc[df['Included']=='Yes']
df["id"] = df["Date"] + "_" + "epoch" + "_" + df["Epoch"].astype(str) + "_" + df["Recorder"].astype(str)
df["id2"] = df["Date"] + "_" + df["Double_cut"].astype(str) + "_" + df["Recorder"].astype(str)
df["ms_id"] = df["Date"] + "_" + df["Double_cut"].astype(str)
df = df.sort_values(by=['Genotype'], ascending=False)
wt_count = df[df["Genotype"] == "WT"].shape[0]
ko_count = df[df["Genotype"] == "KO"].shape[0]
print(f"{wt_count} WT cells and {ko_count} KO cells")
directory = path_data # directory of data folder
columns = [
    "Peak direction",
    "Amplitude",
    "Peak time (ms)",
    "Est_decay (ms)"
]
group = "Genotype"
unique_id = "id"
rec_id = "id2"
sheet = "Final data"

# extract data from sheet "Raw data"
all_dict = get_data(df, directory, columns, group, unique_id, rec_id, sheet)

col_wt = all_dict["Amplitude"]["WT"]["AMPA"].columns
col_ko = all_dict["Amplitude"]["KO"]["AMPA"].columns

df_ei = pd.DataFrame(columns=['id2', 'Genotype', 'ms_id'])
count = 0
for i in col_wt:
    c1 = all_dict["Amplitude"]["WT"]["AMPA"][i].dropna()
    c2 = all_dict["Amplitude"]["WT"]["GABA"][i].dropna()
    c3 = c1/c2
    c4 = all_dict["Est_decay (ms)"]["WT"]["AMPA"][i].dropna()
    c5 = all_dict["Est_decay (ms)"]["WT"]["GABA"][i].dropna()
    c6 = all_dict["Peak time (ms)"]["WT"]["AMPA"][i].dropna()
    c7 = all_dict["Peak time (ms)"]["WT"]["GABA"][i].dropna()
    df_temp = pd.concat([c1,c2,c3,c4,c5,c6,c7], axis=1, ignore_index=True)
    df_ei = pd.concat([df_ei,df_temp], ignore_index=True)
    df_ei.loc[count:(count+c1.size), 'id2'] = i
    df_ei.loc[count:(count+c1.size), 'Genotype'] = "WT"
    count += c1.size

for i in col_ko:
    c1 = all_dict["Amplitude"]["KO"]["AMPA"][i].dropna()
    c2 = all_dict["Amplitude"]["KO"]["GABA"][i].dropna()
    c3 = c1/c2
    c4 = all_dict["Est_decay (ms)"]["KO"]["AMPA"][i].dropna()
    c5 = all_dict["Est_decay (ms)"]["KO"]["GABA"][i].dropna()
    c6 = all_dict["Peak time (ms)"]["KO"]["AMPA"][i].dropna()
    c7 = all_dict["Peak time (ms)"]["KO"]["GABA"][i].dropna()
    df_temp = pd.concat([c1,c2,c3,c4,c5,c6,c7], axis=1, ignore_index=True)
    df_ei = pd.concat([df_ei,df_temp], ignore_index=True)
    df_ei.loc[count:(count+c1.size), 'id2'] = i
    df_ei.loc[count:(count+c1.size), 'Genotype'] = "KO"
    count += c1.size
    
df_ei = df_ei.rename(columns={0: 'AMPA current', 1: 'GABA current', 2: 'E/I', 3: 'AMPA Tau', 4: 'GABA Tau', 5: 'AMPA Peaktime', 6: 'GABA Peaktime'})
df_ei['AMPA Peaktime'] -=1000
df_ei['GABA Peaktime'] -=1000
df_ei.loc[df_ei['GABA Peaktime'] > 50, 'GABA Peaktime'] = np.nan
df_ei["ms_id"] = df_ei['id2'].str[:13]

# MW test
columns = [
    'AMPA current',
    'GABA current',
    'E/I',
    'AMPA Tau',
    'GABA Tau',
    'AMPA Peaktime',
    'GABA Peaktime'
    ]

for j in columns:
    value_wt = df_ei[df_ei['Genotype'] == 'WT'][j].dropna()
    value_ko = df_ei[df_ei['Genotype'] == 'KO'][j].dropna()
    pvalue = mannwhitneyu(value_wt,value_ko)[1]
    print(f"{j}: {string_pvalue(pvalue)}")

# mean and std
for i in ["WT","KO"]:
    for j in columns:
        col_mean = df_ei[df_ei['Genotype'] == i][j].mean()
        col_mean = round(col_mean,1)
        col_std = df_ei[df_ei['Genotype'] == i][j].std()
        col_std = round(col_std,1)
        print(f"{i}_{j}: mean = {col_mean} std = {col_std}")

# Amp v.s. YFP intensity
df_temp = df.dropna(subset=['mCH_intensity'])
df_temp = df_temp.sort_values(by=['Genotype'], ascending=False)

# two-way ANOVA
model = ols('Amplitude_AMPA~Genotype+mCH_intensity+Genotype:mCH_intensity',df_temp).fit()
sm.stats.anova_lm(model, typ=2)


#%% C1V1 control

path_sum = "path of excel file"

df = pd.read_excel(path_sum) # directory of summery excel file
df.drop(df[df["Included"]=="No"].index, inplace=True)
df = df.sort_values(by=['Genotype'], ascending=False)
df1 = df[df["Recording"] == "VC"].reset_index(drop=True)
df2 = df[df["Recording"] == "CC"].reset_index(drop=True)
wt_vc_count = df1[df1["Genotype"] == "WT"].shape[0]
ko_vc_count = df1[df1["Genotype"] == "KO"].shape[0]
wt_cc_count = df2[df2["Genotype"] == "WT"].shape[0]
ko_cc_count = df2[df2["Genotype"] == "KO"].shape[0]
print(f"VC: {wt_vc_count} WT and {ko_vc_count} KO")
print(f"CC: {wt_cc_count} WT and {ko_cc_count} KO")

model1 = ols('Amplitude~Genotype+YFP_soma+Genotype:YFP_soma',df1).fit()
sm.stats.anova_lm(model1, typ=2)

model2 = ols('AP_number~Genotype+YFP_soma+Genotype:YFP_soma',df2).fit()
sm.stats.anova_lm(model2, typ=2)


#%% N/A ratio measrement

def get_data(df, directory, columns, group, unique_id, sheet="Summary"):
    paths = list(Path(directory).rglob("*.xlsx"))
    df_dict = {}
    for i in paths:
        key = f"{i.parts[-3]}_{i.stem}"
        df_dict[key] = pd.read_excel(i, sheet_name=sheet)

    data_dict = {}
    for i in range(df.shape[0]):
        if df[group].iloc[i] in data_dict:
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]
        else:
            data_dict[df[group].iloc[i]] = {}
            if df[unique_id].iloc[i] in df_dict:
                data_dict[df[group].iloc[i]][df[unique_id].iloc[i]] = df_dict[
                    df[unique_id].iloc[i]
                ]

    all_dict = {}
    for k in columns:
        temp_dict = {}
        all_dict[k] = temp_dict
        for i in data_dict.keys():
            temp_dict[i] = {}
            for j in data_dict[i].keys():
                temp_dict[i][j] = data_dict[i][j][k]
            temp_dict[i] = pd.DataFrame(temp_dict[i])
    return all_dict

def string_pvalue(pvalue):
    if pvalue >= 0.01:
        p_print = f"{round(pvalue, 3)}"
    elif pvalue >= 0.001:
        p_print = f"{round(pvalue, 3)}"
    else:
        p_print = f"{pvalue:.2e}"
    return(p_print)

# set path
path_sum = "path of excel file"
path_data = "path of data folder"

df = pd.read_excel(path_sum) # directory of summery excel file
df["id"] = df["Date"] + "_" + "epoch" + "_" + df["Epoch"].astype(str) + "_" + df["Recorder"].astype(str)
df.drop(df[df["Included"]=="No"].index, inplace=True)
df = df.sort_values(by=['Genotype'], ascending=False)
df1 = df[df["Cell type"] == "PYR"].reset_index(drop=True)
df2 = df[df["Cell type"] == "E2"].reset_index(drop=True)
wt_pyr_count = df1[df1["Genotype"] == "WT"].shape[0]
ko_pyr_count = df1[df1["Genotype"] == "KO"].shape[0]
wt_e2_count = df2[df2["Genotype"] == "WT"].shape[0]
ko_e2_count = df2[df2["Genotype"] == "KO"].shape[0]
print(f"{wt_pyr_count} WT PYR and {ko_pyr_count} KO PYR")
print(f"{wt_e2_count} WT E2+ and {ko_e2_count} KO E2+")
directory = path_data # directory of data folder
columns = [
    "Peak direction",
    "Amplitude",
    "Peak time (ms)",
    "Est_decay (ms)"
]
group = "Genotype"
unique_id = "id"
sheet = "Final data"

# extract data from sheet "Raw data"
all_dict_pyr = get_data(df1, directory, columns, group, unique_id, sheet)
all_dict_e2 = get_data(df2, directory, columns, group, unique_id, sheet)

# PYR
col_wt = all_dict_pyr["Amplitude"]["WT"].columns
col_ko = all_dict_pyr["Amplitude"]["KO"].columns

df_na_pyr = pd.DataFrame()
for i, id2 in enumerate(col_wt):
    c1 = all_dict_pyr["Amplitude"]["WT"].iloc[0,i]
    c2 = all_dict_pyr["Amplitude"]["WT"].iloc[1,i]
    c3 = c2/c1
    df_temp = pd.DataFrame([[id2, "WT", c1, c2, c3]])
    df_na_pyr = pd.concat([df_na_pyr,df_temp], axis=0).reset_index(drop=True)

for i, id2 in enumerate(col_ko):
    c1 = all_dict_pyr["Amplitude"]["KO"].iloc[0,i]
    c2 = all_dict_pyr["Amplitude"]["KO"].iloc[1,i]
    c3 = c2/c1

    df_temp = pd.DataFrame([[id2, "KO", c1, c2, c3]])
    df_na_pyr = pd.concat([df_na_pyr,df_temp], axis=0).reset_index(drop=True)

    
df_na_pyr.rename(columns={0: 'id', 1: 'Genotype', 2: 'AMPA Amp', 3: 'NMDA Amp', 4: 'N/A ratio'}, inplace=True)
df_na_pyr['Date'] = df_na_pyr['id'].str[:11]

# E2
col_wt = all_dict_e2["Amplitude"]["WT"].columns
col_ko = all_dict_e2["Amplitude"]["KO"].columns

df_na_e2 = pd.DataFrame()
for i, id2 in enumerate(col_wt):
    c1 = all_dict_e2["Amplitude"]["WT"].iloc[0,i]
    c2 = all_dict_e2["Amplitude"]["WT"].iloc[1,i]
    c3 = c2/c1
    df_temp = pd.DataFrame([[id2, "WT", c1, c2, c3]])
    df_na_e2 = pd.concat([df_na_e2,df_temp], axis=0).reset_index(drop=True)


for i, id2 in enumerate(col_ko):
    c1 = all_dict_e2["Amplitude"]["KO"].iloc[0,i]
    c2 = all_dict_e2["Amplitude"]["KO"].iloc[1,i]
    c3 = c2/c1
    df_temp = pd.DataFrame([[id2, "KO", c1, c2, c3]])
    df_na_e2 = pd.concat([df_na_e2,df_temp], axis=0).reset_index(drop=True)

df_na_e2.rename(columns={0: 'id', 1: 'Genotype', 2: 'AMPA Amp', 3: 'NMDA Amp', 4: 'N/A ratio'}, inplace=True)
df_na_e2['Date'] = df_na_e2['id'].str[:11]

# MW test
value_wt = df_na_pyr[df_na_pyr['Genotype'] == 'WT']['N/A ratio']
value_ko = df_na_pyr[df_na_pyr['Genotype'] == 'KO']['N/A ratio']
pvalue = mannwhitneyu(value_wt,value_ko)[1]
print(f"PYR:N/A ratio: {string_pvalue(pvalue)}")

value_wt = df_na_e2[df_na_e2['Genotype'] == 'WT']['N/A ratio']
value_ko = df_na_e2[df_na_e2['Genotype'] == 'KO']['N/A ratio']
pvalue = mannwhitneyu(value_wt,value_ko)[1]
print(f"E2:N/A ratio: {string_pvalue(pvalue)}")

# mean and std
for i in ["WT","KO"]:
    col_mean = df_na_pyr[df_na_pyr['Genotype'] == i]['N/A ratio'].mean()
    col_mean = round(col_mean,1)
    col_std = df_na_pyr[df_na_pyr['Genotype'] == i]['N/A ratio'].std()
    col_std = round(col_std,1)
    print(f"PYR-{i}_N/A ratio: mean = {col_mean} std = {col_std}")

for i in ["WT","KO"]:
    col_mean = df_na_e2[df_na_e2['Genotype'] == i]['N/A ratio'].mean()
    col_mean = round(col_mean,1)
    col_std = df_na_e2[df_na_e2['Genotype'] == i]['N/A ratio'].std()
    col_std = round(col_std,1)
    print(f"E2-{i}_N/A ratio: mean = {col_mean} std = {col_std}")


#%% HCN current analysis
# measure Amp of individual acq
start_acq = "start_acq of one Epoch"
end_acq = "end_acq of one Epoch"
paths = []

for i in range(start_acq, end_acq+1):
    paths.append(
        Path("path of data folder")
    ) # directory of acq


exp_manager = ExpManager()
exp_manager.create_exp("filter", paths)
acqs = exp_manager.exp_dict["filter"]

output = {}
for keys, values in acqs.items():
    y = values.array
    y_series = pd.Series(y)
    q80_100 = y_series.iloc[:6000].rolling(window=100, min_periods=1, center=True).quantile(0.8)
    q20_100 = y_series.iloc[:6000].rolling(window=100, min_periods=1, center=True).quantile(0.2)
    q80_1000 = y_series.iloc[6000:].rolling(window=1000, min_periods=1, center=True).quantile(0.8)
    q20_1000 = y_series.iloc[6000:].rolling(window=1000, min_periods=1, center=True).quantile(0.2)
    q80 = pd.concat([q80_100, q80_1000], ignore_index=True)
    q20 = pd.concat([q20_100, q20_1000], ignore_index=True)
    q_mean = (q80 + q20)/2
    
    """
    onset ramge 4000:6000
    end range 13000:22500 for PYR, 6500:10000 for PV
    end range can be manually modified to avoid unexpected drop
    """
    
    onset_x = q_mean[4000:6000].argmax()+4000 # set range for finding onset
    # end_x = q_mean[13000:22500].argmin()+13000 # set range for finding end for PYR
    end_x = q_mean[6500:10000].argmin()+6500 # set range for finding end for PV
    onset_y = y[onset_x-500:onset_x+500].mean()
    end_y = y[end_x-500:end_x+500].mean()
    amp = onset_y - end_y
    amp = 0 if amp < 0 else amp
    
    temp = {
        "onset_y": onset_y,
        "onset_x": onset_x,
        "end_y":  end_y,
        "end_x": end_x,
        "amplitude": amp
    }

    output[keys] = temp

steps = 10

keys = ['onset_y', 'onset_x', 'end_y', 'end_x', 'amplitude']

if len(output) % steps ==0:
    n_groups = len(output) // steps # number of cycles
else: # if ouput miss few acq for a complete cycle, create dummy acq
    n_required = steps - (len(output) % steps) 
    dummy_data = {key: np.nan for key in keys}
    for i in range(n_required):
        output[f"dummy_{i}"] = dummy_data
    n_groups = len(output) // steps

dfs = {} # dfs contains raw data of one epoch

for key in keys:
    group_data = []
    for group_idx in range(n_groups):
        group = [
            list(output.values())[i][key] 
            for i in range(group_idx * steps, (group_idx + 1) * steps)
        ]
        group_data.append(group)
    
    dfs[key] = pd.DataFrame(np.array(group_data).T, columns=[f'cycle{j+1}' for j in range(n_groups)])

for key in keys:
    dfs[key]['mean'] = dfs[key].mean(axis=1)
    
df_mean = pd.DataFrame({key: dfs[key]['mean'] for key in keys})

# perform stats
path_sum = "path of excel file"
df = pd.read_excel(path_sum) # directory of summery excel file
df.drop(df[df["Included"]=="No"].index, inplace=True)
df = df.sort_values(by=['Genotype'], ascending=False)
df1 = df[df["Cell type"] == "PYR"].reset_index(drop=True)
df2 = df[df["Cell type"] == "E2"].reset_index(drop=True)
wt_pyr_count = df1[df1["Genotype"] == "WT"].shape[0]
ko_pyr_count = df1[df1["Genotype"] == "KO"].shape[0]
wt_e2_count = df2[df2["Genotype"] == "WT"].shape[0]
ko_e2_count = df2[df2["Genotype"] == "KO"].shape[0]
print(f"{wt_pyr_count} WT PYR and {ko_pyr_count} KO PYR")
print(f"{wt_e2_count} WT E2+ and {ko_e2_count} KO E2+")

columns = [
    "Max_Amp",
    "Max_Current_density",
    "tau",
]

# MW test
for j in columns:
    value_wt = df1[df1['Genotype'] == 'WT'][j].dropna()
    value_ko = df1[df1['Genotype'] == 'KO'][j].dropna()
    pvalue = mannwhitneyu(value_wt,value_ko)[1]
    print(f"PYR:-{j}: {string_pvalue(pvalue)}")
    
for j in columns:
    value_wt = df2[df2['Genotype'] == 'WT'][j].dropna()
    value_ko = df2[df2['Genotype'] == 'KO'][j].dropna()
    pvalue = mannwhitneyu(value_wt,value_ko)[1]
    print(f"E2:-{j}: {string_pvalue(pvalue)}")

# mean and std
for i in ["WT","KO"]:
    for j in columns:
        col_mean = df1[df1['Genotype'] == i][j].mean()
        col_mean = round(col_mean,1)
        col_std = df1[df1['Genotype'] == i][j].std()
        col_std = round(col_std,1)
        print(f"PYR: {i}_{j}: mean = {col_mean} std = {col_std}")

for i in ["WT","KO"]:
    for j in columns:
        col_mean = df2[df2['Genotype'] == i][j].mean()
        col_mean = round(col_mean,1)
        col_std = df2[df2['Genotype'] == i][j].std()
        col_std = round(col_std,1)
        print(f"E2: {i}_{j}: mean = {col_mean} std = {col_std}")

# Vm-Id
df = pd.read_excel(path_sum, sheet_name="current density")
df.drop(df[df["Included"]=="No"].index, inplace=True)
df = df.sort_values(by=['Genotype'], ascending=False)
df3 = df[df["Cell type"] == "PYR"].reset_index(drop=True)
df4 = df[df["Cell type"] == "E2"].reset_index(drop=True)

df3_melted = df3.melt(id_vars='Genotype', value_vars=[-70, -80, -90, -100, -110, -120, -130, -140, -150, -160], var_name='Vm', value_name='current_density')
df3_melted = df3_melted.dropna(subset=['current_density'])
df3_summary = df3_melted.groupby(['Genotype', 'Vm'])['current_density'].agg(['mean', sem]).reset_index()

df4_melted = df4.melt(id_vars='Genotype', value_vars=[-70, -80, -90, -100, -110, -120, -130, -140, -150, -160], var_name='Vm', value_name='current_density')
df4_melted = df4_melted.dropna(subset=['current_density'])
df4_summary = df4_melted.groupby(['Genotype', 'Vm'])['current_density'].agg(['mean', sem]).reset_index()

# two way ANOVA
model = ols('current_density~Genotype+Vm+Genotype:Vm',df3_melted).fit()
sm.stats.anova_lm(model, typ=2)
model = ols('current_density~Genotype+Vm+Genotype:Vm',df4_melted).fit()
sm.stats.anova_lm(model, typ=2)

#%% curve fit function for HCN
# Define sigmoid function
def sigmoid(t, A, B, t_half, k):
    return A + B / (1 + np.exp((t - t_half) / k))

# Define exponential decay function
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

# Main function
def fit_decay(df, fit_type='double_exponential', plot=True):
    # Read data
    I_raw = df.iloc[:, 0].values
    t_full = np.arange(len(I_raw)) * 0.1  # Assuming 10 kHz → 0.1 ms interval

    # --- Automatically find peak to baseline section ---
    peak_idx = np.argmax(I_raw)       # Index of maximum value
    baseline_idx = len(I_raw) - 1     # Index of baseline

    I_seg = I_raw[peak_idx:baseline_idx+1]
    t_seg = t_full[peak_idx:baseline_idx+1]

    # Flip data to match sigmoid model
    I_flip = -I_seg

    if fit_type == 'sigmoid':
        # --- Sigmoid fitting ---
        p0 = [min(I_flip), max(I_flip) - min(I_flip), t_seg[np.argmin(I_flip)], 100]
        popt, _ = curve_fit(sigmoid, t_seg, I_flip, p0=p0, maxfev=10000)
        I_fit = sigmoid(t_seg, *popt)

        Amplitude = -popt[1]  # Amplitude
        k = popt[3]  # Slope (k)

        I_fit_real = -I_fit

        print("🔵 Sigmoid curve fit")
        print(f"RMSE: {np.sqrt(mean_squared_error(I_seg, I_fit_real)):.4f}")
        print(f"R²: {r2_score(I_seg, I_fit_real):.4f}")
        print(f"Amplitude: {Amplitude:.4f}")
        print(f"k: {k:.4f}") # bigger k means flatter

    elif fit_type == 'exponential':
        # --- Single exponential fitting ---
        p0_exp = [I_seg[0] - I_seg[-1], 100, I_seg[-1]]
        popt_exp, _ = curve_fit(exp_decay, t_seg, I_seg, p0=p0_exp)
        I_fit_exp = exp_decay(t_seg, *popt_exp)

        tau_eff = popt_exp[1]
        I_fit_real = I_fit_exp

        print("🔵 Exponential curve fit")
        print(f"RMSE: {np.sqrt(mean_squared_error(I_seg, I_fit_real)):.4f}")
        print(f"R²: {r2_score(I_seg, I_fit_real):.4f}")
        print(f"τ_eff: {tau_eff:.4f} ms")

    elif fit_type == 'double_exponential':
        # --- Double exponential fitting ---
        def double_exp(t, A1, tau1, A2, tau2, C):
            return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

        p0_double = [I_seg[0] - I_seg[-1], 5, (I_seg[0] - I_seg[-1]) / 2, 50, I_seg[-1]]
        # popt_double, _ = curve_fit(double_exp, t_seg, I_seg, p0=p0_double)
        popt_double, _ = curve_fit(
            double_exp,
            t_seg,
            I_seg,
            p0=p0_double,
            bounds=([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
            )
        
        A1, tau1, A2, tau2, C = popt_double

        # Separate components
        I_fast = A1 * np.exp(-t_seg / tau1)
        I_slow = A2 * np.exp(-t_seg / tau2)
        I_fit_double = I_fast + I_slow + C

        tau_eff = min(tau1, tau2)
        I_fit_real = I_fit_double

        print("🔵 Double Exponential curve fit")
        print(f"RMSE: {np.sqrt(mean_squared_error(I_seg, I_fit_real)):.4f}")
        print(f"R²: {r2_score(I_seg, I_fit_real):.4f}")
        print(f"Tau1 = {tau1:.4f} ms")
        print(f"Tau2 = {tau2:.4f} ms")
        print(f"τ_eff: {tau_eff:.4f} ms")

    else:
        raise ValueError("Invalid fit_type. Use 'sigmoid', 'exponential', or 'double_exponential'.")

    # --- Calculate evaluation metrics ---
    rmse = np.sqrt(mean_squared_error(I_seg, I_fit_real))
    r2 = r2_score(I_seg, I_fit_real)

    # --- Display plot ---
    if plot is not False:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(t_full, I_raw, label='Raw Current', color='black', alpha=0.5)
        ax.plot(t_seg, I_fit_real, '--r', label=f'Fitted {fit_type.capitalize()}')
        ax.scatter(t_full[peak_idx], I_raw[peak_idx], color='blue', zorder=5, label="Peak")

        if fit_type == 'double_exponential':
            # Plot fast and slow components
            ax.plot(t_seg, I_fast + C, '--', color='orange', label='Fast Component')
            ax.plot(t_seg, I_slow + C, '--', color='green', label='Slow Component')

            # Tau markers
            tau1_time = peak_idx + int(tau1 / 0.1)
            tau2_time = peak_idx + int(tau2 / 0.1)

            if tau1_time < len(I_raw):
                ax.scatter(t_full[tau1_time], I_raw[tau1_time], color='orange', label=f'Tau1 = {tau1:.4f} ms')
            if tau2_time < len(I_raw):
                ax.scatter(t_full[tau2_time], I_raw[tau2_time], color='green', label=f'Tau2 = {tau2:.4f} ms')

        elif fit_type == 'exponential':
            tau_time = peak_idx + int(tau_eff / 0.1)
            if tau_time < len(I_raw):
                ax.scatter(t_full[tau_time], I_raw[tau_time], color='green', label=f'τ_eff: {tau_eff:.4f} ms')

        # Custom formatting
        ax.set_xlim(0, 1900)
        ax.set_xlabel('Time (ms)', fontsize=20)
        ax.set_ylabel('Ih current (pA)', fontsize=20)
        ax.set_xticks([0, 500, 1000, 1500])
        ax.set_xticklabels(['0', '500', '1000', '1500'], fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20, bottom=True, left=True, width=2)

    if fit_type == 'double_exponential':
        return rmse, r2, tau1, tau2
    else:
        return rmse, r2, tau_eff if fit_type == 'exponential' else k


#%% plot ms avg

savefig = True
save_path = "path for save"
filename = "file name"
column = "item for analysis"
ylabel = "y label"
ylim = [None,None] # determine ylim
decimals = 1 # determine decimals
title = 'title'

temp = (
    CategoricalPlot(data='df of data')
    .grouping(
        # subgroup="Date",
        # subgroup_order=[r"WT", r"KO"],
        group="Genotype",
        group_order=[r"WT", r"KO"],
        group_spacing=0.8,
    )
    # .violin(facecolor="none", width=0.8, showmeans=False)
    .jitteru(
        unique_id="Date",
        edgecolor="none",
        marker="o",
        width=0.8,
        markersize=10,
        alpha=0.5,
        color={
            "WT": "gray",
            "KO": "red",
        },
    )
    .jitteru(
        unique_id="Date",
        edgecolor="none",
        marker="X",
        width=0.8,
        markersize=15,
        alpha=0.9,
        color={
            "WT": "black",
            "KO": "darkred",
        },
        agg_func="mean",
    )
    .summary(
        func="mean",
        # agg_func="mean",
        # unique_id="id",
        capsize=6,
        capstyle="round",
        barwidth=0.8,
        err_func="ci",
        linewidth=2,
        color="black",
    )
    .axis_format(ysteps=5, linewidth=0.5, tickwidth=0.5)
    .labels(labelsize=22, ticklabel_size=20, font="Arial")
    .axis(ylim=ylim, ydecimals=decimals)

    .plot_data(y=column, ylabel=ylabel, title=title)
    .figure(figsize=(6,6))
    .plot(
        savefig=savefig,
        path=save_path,
        filetype="svg",
        backend="matplotlib",
        filename=filename,
    )
)

