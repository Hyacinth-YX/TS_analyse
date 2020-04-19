import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib as mlt

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False



# 获取数据的主入口
def GetDataFromExcel(filepath,sheet):
    data = pd.read_excel(filepath,sheet_name=sheet)
    data = DataClean (data)
    return data

# 在指定画布大小上画出各个数据的箱线图判断离群值
def CheckDataValition(data,x,y):
    index = 1
    for col in data.columns.tolist():
        if col == 'time':
            pass
        else:
            plt.subplot(x,y,index)
            pd.DataFrame(data[col]).boxplot()
            index += 1
    plt.show ()

# 对每一列的数据，根据箱线图上下限把异常值转化为上下限
def BoxPlotFill(col):
    # 计算iqr
    iqr = col.quantile(0.75)-col.quantile(0.25)
    # 异常值栅栏
    up_num = col.quantile(0.75) + 1.5 * iqr
    low_num = col.quantile(0.25) - 1.5 * iqr
    # 转换函数：大于上界的用上界代替，下界亦然
    def BoxTrans(num):
        if num > up_num:
            return up_num
        elif num < low_num:
            return low_num
        else:
            return num
    a = col.apply(BoxTrans)
    return a

# 生成箱线图验证有效性并清洗离群值
def DataClean(data):
    CheckDataValition (data, 4, 4)
    for col in data.columns.tolist():
        if col == 'time':
            continue
        try:
            data[col] = BoxPlotFill(data[col])
        except Exception as e:
            print(e)
    CheckDataValition (data, 4, 4)
    return data

# 将"Q199"按照解读方式替换为"1999/1/1"
def TimeStampTrans(str_num):
    if len(str_num) != 4:
        print("unformal data")
        return str_num
    else:
        year = "1999" if str_num[-2:]=='99' else "20" + str_num[-2:]
        month = str((int(str_num[1]) - 1) * 3 + 1)
        string = year + '/' + month + '/' + '1'
        time = pd.to_datetime(year + '/' + month + '/' + '1')
        return time



if __name__ == '__main__':
    # # 源数据处理
    # filepath = './附件1数据.xlsx'
    # sheet = 1
    # data = GetDataFromExcel(filepath,sheet)
    # # 替换时间戳
    # data['time'] = pd.to_datetime (data['time'].apply (TimeStampTrans))
    # # 保存处理好的数据
    # data.to_csv("./附件1数据完成清理.csv",index=False,mode='w')
    # --------------
    # 从保存好的数据开始处理数据
    data = pd.read_csv("附件1数据完成清理.csv", index_col=0, parse_dates=[0])
    # 去除空值所在行
    data_month = data['北美整体规模']
    isnull = data_month.isnull()
    null_index = data_month[isnull].index.tolist()
    data_month = data_month.drop(null_index)
    # 进行重采样，对于空值采取线性插值
    data_month = data_month.resample('3M').mean().interpolate('linear')
    # 构建训练集(15年作为训练集 5年作为测试集)
    data_train = data_month['1999':'2018']
    data_test = data_month['2018':'2019']
    # # 观察训练集趋势
    # data_train.plot(figsize=(12,8))
    # plt.legend(bbox_to_anchor=(1.25,0.5))
    # plt.title('12 north america')
    # sns.despine()
    # plt.show()
    # 做差分增加平稳性
    d=2
    if d == 1:
        data_diff = data_train.diff ().dropna ()
    elif d == 2:
        data_diff = data_train.diff ().dropna ().diff ().dropna ()
    else:
        data_diff = data_train.diff ().dropna ().diff ().dropna ().diff ().dropna ()

    ## 观察差分情况
    plt.figure()
    plt.plot(data_diff)
    plt.title(str(d) + 'period diff')
    plt.show()

    # 绘制 acf 和 pcf来判断参数
    acf = plot_acf(data_diff, lags=20)
    plt.title('ACF')
    acf.show()
    pacf = plot_pacf(data_diff,lags=20)
    plt.title("PACF")
    pacf.show()

    q = int(input("please input q:"))
    p = int(input("please input p:"))


    # 训练模型
    model = ARIMA(data_train,order=(p,d,q),freq='3M')
    result = model.fit()

    pred = result.predict('20181031','20191031',dynamic=True,typ='levels')
    # print(pred)
    plt.figure(figsize=(6,6))
    plt.xticks(rotation=45)
    plt.plot(pred)
    plt.plot(data_train)
    plt.plot(data_test)
    plt.title('3G North America')
    plt.show()
