import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

def comp_price_extractor(df):
    '''Creates two csvs to be used as new variables in var_clean'''
    company_price = df.groupby('Company')['Price_euros'].describe()
    company_price = company_price['50%'].reset_index()
    company_price.rename(columns={'50%':'company_price'},inplace=True)
    type_price = df.groupby('TypeName')['Price_euros'].describe()
    type_price = type_price['50%'].reset_index()
    type_price.rename(columns={'50%':'type_price'},inplace=True)
    company_price.to_csv('data/auxiliar/company_price.csv',index_label=False)
    type_price.to_csv('data/auxiliar/type_price.csv',index_label=False)


def var_clean(df,df_gpus,df_cpus,df_company,df_type):
    ### Limpieza básica de variables
    df['Ram'] = df['Ram'].str.replace("GB","").astype(int)
    df['Weight'] = df['Weight'].str.replace("kg","").astype(float)
    ### Asignamos el precio mediano del portátil por su marca
    df = pd.merge(df,df_company, on='Company',how = 'left')
    ### Asignamos el precio mediano del portátil por su tipo
    df = pd.merge(df,df_type, on='TypeName',how = 'left')
    df.rename(columns={'50%':'type_price'},inplace=True)
    ### Valor numérico de píxeles
    df['screen_val'] = df['ScreenResolution'].str.extract(r'(\d+(?=x))').astype(float) * df['ScreenResolution'].str.extract(r'((?<=x)\d+)').astype(float)
    ### Dummies de tipos de pantallas
    df['ips'] = np.where(df['ScreenResolution'].str.contains('IPS'),1,0).astype(int)
    df['touchscreen'] = np.where(df['ScreenResolution'].str.contains('Touchscreen'),1,0).astype(int)
    ### Cpu: ghz como float
    df['ghz'] = df['Cpu'].str.extract(r'(\d+\.\d+(?=GHz))').astype(float)
    ### Cpu: marca
    df['intel'] = np.where(df['Cpu'].str.contains('Intel'),1,0).astype(int)
    df['amd'] = np.where(df['Cpu'].str.contains('AMD'),1,0).astype(int)
    ### Cpu: limpieza para merge
    ### Port: CPU limpieza para merge y merge
    df[['Cpu', 'frequency']] = df['Cpu'].str.extract(r'(.+)\s(\d+\.\d+GHz)')
    df['Cpu'] = df['Cpu'].str.lower()
    df.drop(columns=['frequency'],inplace=True)
    df = pd.merge(df,df_cpus, how='left',left_on='Cpu',right_on='name')
    ### Gpu: limpieza para merge, y merge de df_gpus:
    df['Gpu'] = df['Gpu'].str.lower().str.replace("nvidia ","").str.replace("amd ","").str.replace('intel hd graphics','intel hd').str.replace('intel iris plus graphics','intel iris plus')
    df = pd.merge(df,df_gpus, how='left',left_on='Gpu',right_on='name')
    df['gpu_score'] = np.where((df['Gpu'].str.contains('intel hd')) & (df['gpu_score'].isna()),200,df['gpu_score'])
    df['cpu_score'].fillna(df['cpu_score'].mean(),inplace=True)
    ### OS: limpieza
    df['OpSys'] = df['OpSys'].str.lower()
    df['linux']=np.where(df['OpSys'].str.contains('linux'),1,0).astype(int)
    df['windows']=np.where(df['OpSys'].str.contains('windows'),1,0).astype(int)
    df['mac']=np.where(df['OpSys'].str.contains('mac'),1,0).astype(int)
    df['no os']=np.where(df['OpSys'].str.contains('no os'),1,0).astype(int)
    ### Discos duros:
    df['Memory'] = df['Memory'].str.lower()
    df['Memory'] = df['Memory'].str.replace("tb","000")
    df['Memory'] = df['Memory'].str.replace("gb","")
    df['ssd']=np.where(df['Memory'].str.contains('ssd'),1,0).astype(int)
    df['hdd']=np.where(df['Memory'].str.contains('hdd'),1,0).astype(int)
    df['Memory'] = df['Memory'].str.extract('(\d+)').astype(int)
    ###Rellenado de nans
    df['ghz'].fillna(df['ghz'].mean(),inplace=True)
    df['gpu_score'].fillna(df['gpu_score'].mean(),inplace=True)
    return df



def selen_downl_cpu():
    ###TODO: Terminar de encapsular bien la función, que coja el "all" bien
    '''Webscraps data for benchmarks. Used for:
        https://www.cpubenchmark.net/CPU_mega_page.html
        https://www.videocardbenchmark.net/GPU_mega_page.html
    '''
    ### WARNING: Selection of All entries is manual!
    service = Service(executable_path='./chromedriver/chromedriver.exe')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.cpubenchmark.net/CPU_mega_page.html")
    loadMore = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/button[2]')
    loadMore.click()
    entries = Select(driver.find_element(By.CSS_SELECTOR, '#cputable_length > label > select'))
    entries.selectByvalue("All")
    cpu_scoring = {'cpu_name':[],'cpu_score':[]}
    i=1

    while i < 5462:
        print("Iteración",i)
        path = driver.find_element(By.XPATH, f'/html/body/div[4]/div/div/div[3]/div[2]/div[2]/div[1]/table/tbody/tr[{i}]/td[2]')
        cpu_scoring['name'].append(path.text)
        path2 = driver.find_element(By.XPATH, f'/html/body/div[4]/div/div/div[3]/div[2]/div[2]/div[1]/table/tbody/tr[{i}]/td[4]')
        cpu_scoring['score'].append(path2.text)
        i+=1

    df_cpu = pd.DataFrame(cpu_scoring)
    df_cpu['score'] = df_cpu['score'].str.replace(',','').astype(float)
    pd.DataFrame(df_cpu).to_csv('data/auxiliar/cpu_scoring.csv',index_label=False)

def selen_downl_gpu():
    service = Service(executable_path='./chromedriver/chromedriver.exe')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.videocardbenchmark.net/GPU_mega_page.html")
    loadMore = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/button[2]')
    loadMore.click()
    entries = Select(driver.find_element(By.CSS_SELECTOR, '#cputable_length > label > select'))
    entries.selectByvalue("All")
    gpu_scoring = {'gpu_name':[],'gpu_score':[]}
    i=1

    while i < 2695:
        path = driver.find_element(By.XPATH, f'/html/body/div[4]/div/div/div[3]/div[2]/div[2]/div[1]/table/tbody/tr[{i}]/td[2]')
        gpu_scoring['name'].append(path.text)
        path2 = driver.find_element(By.XPATH, f'/html/body/div[4]/div/div/div[3]/div[2]/div[2]/div[1]/table/tbody/tr[{i}]/td[3]')
        gpu_scoring['score'].append(path2.text)
        i+=1

    df_gpu = pd.DataFrame(gpu_scoring)
    df_gpu['name'] = df_gpu['name'].str.lower()
    df_gpu['score'] = df_gpu['score'].str.replace(',','').astype(float)
    pd.DataFrame(df_gpu).to_csv('data/auxiliar/gpu_scoring.csv',index_label=False)

def gpu_cpu_clean(df_gpus,df_cpus):
    df_gpus['name'] = df_gpus['name'].str.lower().str.replace('intel hd graphics','intel hd').str.replace('intel iris plus graphics','intel iris plus')
    df_cpus['name'] = df_cpus['name'].str.lower()
    df_cpus['name'] = df_cpus['name'].str.replace("-"," ")
    df_cpus[['name','ghz']] = df_cpus['name'].str.split('@', expand=True,n=2)
    df_cpus['name'] = df_cpus['name'].str.rstrip()
    df_cpus.rename(columns={'score': 'cpu_score'},inplace=True)
    df_cpus.drop(columns=['ghz'],inplace=True)
    df_gpus.to_csv('data/auxiliar/gpu_clean.csv',index_label=False)
    df_cpus.to_csv('data/auxiliar/cpu_clean.csv',index_label=False)

def lin_reg(df,cross=True,scaler=True,polim=True,deg=2,store_val=True):
    x1=df.drop(columns= ['Price_euros']).copy()
    y1=df['Price_euros']
    if polim == True:
        pol_feats = PolynomialFeatures(degree = deg)
        pol_feats.fit(x1)
        x1 = pol_feats.transform(x1)
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size=.2, random_state=42)
    if scaler == True:
        scaler = preprocessing.StandardScaler()
        scaler.fit(x1_train)
        x1_train = scaler.transform(x1_train)
        x1_test = scaler.transform(x1_test)
    model = LinearRegression()
    model.fit(x1_train,y1_train)
    pred1 = model.predict(x1_test)
    print('MAE:', metrics.mean_absolute_error(y1_test, pred1))
    print('RMSE:', metrics.root_mean_squared_error(y1_test, pred1))
    print('='*20)
    print('r2 train',model.score(x1_train,y1_train))
    print('r2 test',model.score(x1_test,y1_test))
    if cross == True:
        kfold = model_selection.KFold(n_splits=5)
        cv_results = model_selection.cross_val_score(model, x1_train, y1_train, cv=kfold)
        print('='*20)
        print('Cross validation results:')
        print(cv_results.mean(), cv_results.std())
    if store_val == True and polim == True:
        return model, x1_train, y1_train, x1_test, y1_test,pol_feats,scaler
    if store_val == True:
        return model, x1_train, y1_train, x1_test, y1_test,scaler