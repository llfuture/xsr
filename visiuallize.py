import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def get_diff(df1, df2):
    df3 = []
    for i in range(len(df1)):
        if i % 100 == 0:
            print(i)
        each1 = df1.iloc[i,:]
        # print(each1)
        isin = False
        for j in range(len(df2)):
            each2 = df2.iloc[j, :]
            # print(each2)
            if each1.equals(each2):
                isin = True
                break
        if not isin:
            df3.append(each1)
    df3 = pd.DataFrame(df3, columns=df1.columns)
    df3 = df3.reindex()
    return df3


def get_visual_data(dir_name):
    all_data_name = set()
    for each in os.listdir(dir_name):
        if each.endswith("csv"):
            each = each[:each.find("3_")]
        # if each not in all_data_name:
            all_data_name.add(each + "3")
    all_data_name = list(all_data_name)
    print(all_data_name)

    for each in all_data_name:
        print("processing file -------------- " + each)
        origin_train = os.path.join(dir_name, each + "_origin_train.csv")
        origin_test = os.path.join(dir_name, each + "_origin_test.csv")
        oversample_train = os.path.join(dir_name, each + "_oversample_train.csv")
        undersample_train = os.path.join(dir_name, each + "_undersample_train.csv")
        origin_train_data = pd.read_csv(origin_train, index_col=0)
        print(origin_train_data.info())
        origin_test_data = pd.read_csv(origin_test, index_col=0)
        oversample_train_data = pd.read_csv(oversample_train, index_col=0)
        undersample_train_data = pd.read_csv(undersample_train, index_col=0)

        print(oversample_train_data.info())

        # 获取origin_train与undersample_train的差集，也就是被删除的数据
        # temp1 = pd.concat([origin_train_data, undersample_train_data])
        # temp1 = temp1.drop_duplicates(keep=False)
        # print(temp1)


        # removed_train_data = origin_train_data[~origin_train_data.isin(undersample_train_data)]
        # removed_train_data = removed_train_data.dropna()
        # print(~origin_train_data.isin(undersample_train_data))
        # print(removed_train_data)

        # 获取oversample_train与origin_train的差集，也就是被增加的数据
        # temp2 = pd.concat([origin_train_data, oversample_train_data])
        # temp2 = temp2.drop_duplicates(keep=False)
        # print(~oversample_train_data.isin(origin_train_data))
        # added_train_data = oversample_train_data[~oversample_train_data.isin(origin_train_data)]
        # print(added_train_data)
        # added_train_data = added_train_data.dropna()
        # added_train_data = temp2[~temp2.isin(origin_train_data)]
        removed_train_data = get_diff(origin_train_data, undersample_train_data)
        added_train_data = get_diff(oversample_train_data, origin_train_data)
        # print(added_train_data)

        # removed_train_data.iloc[:,-1] = removed_train_data.iloc[:,-1].apply(lambda x: 2 if x == 0 else 3)
        # added_train_data.iloc[:,-1] = added_train_data.iloc[:,-1].apply(lambda x: 2 if x == 0 else 3)
        removed_train_data.iloc[:,-1] += 2
        added_train_data.iloc[:,-1] += 4

        removed_train_data.to_csv(os.path.join(dir_name, each + "_removed_train.csv"))
        added_train_data.to_csv(os.path.join(dir_name, each + "_added_train.csv"))

        all_data = pd.concat([origin_train_data, removed_train_data, added_train_data], axis=0)
        all_data.to_csv(os.path.join(dir_name, each + "_all_train.csv"))

    return all_data_name
    # return all_data, origin_train_data, removed_train_data, added_train_data

def visualize(data = None, fname = None):
    if data is not None:
        pass
    else:
        data = pd.read_csv(fname)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(x)
    X_tsne_data = np.vstack((X_tsne.T, y)).T
    print(X_tsne_data)
    df_tsne = pd.DataFrame(X_tsne_data, columns=['X', 'Y', 'label'])
    df_tsne.head()
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df_tsne, hue='label', x='X', y='Y')
    plt.title(parse_fname(fname))
    plt.legend(loc='best')
    plt.savefig(fname + '.pdf')
    plt.show()
##未完成
# def visualize3d(data = None, fname = None):
#     if data is not None:
#         pass
#     else:
#         data = pd.read_csv(fname)
#     x = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#     tsne = TSNE(n_components=3)
#
#     X_tsne = tsne.fit_transform(x)
#     X_tsne_data = np.vstack((X_tsne.T, y)).T
#     print(X_tsne_data)
#     df_tsne = pd.DataFrame(X_tsne_data, columns=['X', 'Y', 'label'])
#     df_tsne.head()
#     plt.figure(figsize=(6, 6))
#     sns.scatterplot(data=df_tsne, hue='label', x='X', y='Y')
#     plt.title(parse_fname(fname))
#     plt.legend(loc='best')
#     plt.savefig(fname + '.pdf')
#     plt.show()

#将文件名转为标准数据名 如，GO_PPI_DM-BP+CC-threshold-3_all_train
#转为 D.M BP.CC over GO and PPI
def parse_fname(fname):
    r = ""
    if 'CE' in fname:
        r += "C.elegans"
    elif 'DM' in fname:
        r += 'D.melanogaster'
    elif 'MM' in fname:
        r += 'M.musculus'
    elif 'SC' in fname:
        r += 'S.cerevisiae'
    else:
        return r
    r += " ("
    if 'BP' in fname:
        r += ' BP'
    if 'CC' in fname:
        r += ' CC'
    if 'MF' in fname:
        r += ' MF'
    r += " ) over GO and PPI"
    return r

def visual_all(dir_name, all_data_name):
    for each in all_data_name:
        eachf = os.path.join(dir_name, each)
        eachfile = eachf + "_all_train.csv"
        eachdata= pd.read_csv(eachfile)

        # eachdata.loc[-1].astype(np.int32)
        # eachdata.iloc[:, -1] = np.where(eachdata.iloc[:, -1] == 0, "Original Negative Instances", eachdata.iloc[:, -1])
        # print(eachdata)
        # eachdata.iloc[:, -1] = np.where(eachdata.iloc[:, -1] == 1, "Original Positive Instances", eachdata.iloc[:, -1])
        # eachdata.iloc[:, -1] = np.where(eachdata.iloc[:, -1] == 2, "Removed Negative Instances", eachdata.iloc[:, -1])
        # eachdata.iloc[:, -1] = np.where(eachdata.iloc[:, -1] == 3, "Removed Positive Instances", eachdata.iloc[:, -1])
        # eachdata.iloc[:, -1] = np.where(eachdata.iloc[:, -1] == 4, "Added Negative Instances", eachdata.iloc[:, -1])
        # eachdata.iloc[:, -1] = np.where(eachdata.iloc[:, -1] == 5, "Added Positive Instances", eachdata.iloc[:, -1])
        eachdata.iloc[eachdata.iloc[:,-1] == 0, -1] = 'Original Negative Instances'
        eachdata.iloc[eachdata.iloc[:, -1] == 1, -1] = 'Original Positive Instances'
        eachdata.iloc[eachdata.iloc[:, -1] == 2, -1] = 'Removed Negative Instances'
        eachdata.iloc[eachdata.iloc[:, -1] == 3, -1] = 'Removed Positive Instances'
        eachdata.iloc[eachdata.iloc[:, -1] == 4, -1] = 'Added Negative Instances'
        eachdata.iloc[eachdata.iloc[:, -1] == 5, -1] = 'Added Positive Instances'
        print(eachdata)
        visualize(eachdata, eachf)

## all_data_name = ['GO_PPI_DM-BP+CC+MF-threshold-3', 'GO_PPI_CE-BP+CC+MF-threshold-3', 'GO_PPI_SC-MF-threshold-3', 'GO_PPI_SC-CC-threshold-3', 'GO_PPI_DM-MF-threshold-3', 'GO_PPI_CE-CC-threshold-3', 'GO_PPI_CE-MF-threshold-3', 'GO_PPI_SC-BP+MF-threshold-3', 'GO_PPI_CE-CC+MF-threshold-3', 'GO_PPI_DM-CC+MF-threshold-3', 'GO_PPI_CE-BP+CC-threshold-3', 'GO_PPI_MM-BP+MF-threshold-3', 'GO_PPI_DM-BP+MF-threshold-3', 'GO_PPI_MM-CC-threshold-3', 'GO_PPI_MM-BP+CC-threshold-3', 'GO_PPI_DM-BP-threshold-3', 'GO_PPI_MM-CC+MF-threshold-3', 'GO_PPI_CE-BP+MF-threshold-3', 'GO_PPI_SC-BP+CC-threshold-3', 'GO_PPI_CE-BP-threshold-3', 'GO_PPI_SC-CC+MF-threshold-3', 'GO_PPI_DM-CC-threshold-3', 'GO_PPI_MM-BP-threshold-3', 'GO_PPI_SC-BP-threshold-3', 'GO_PPI_MM-MF-threshold-3', 'GO_PPI_SC-BP+CC+MF-threshold-3', 'GO_PPI_MM-BP+CC+MF-threshold-3', 'GO_PPI_DM-BP+CC-threshold-3']

# all_data_name = get_visual_data("final_data")
# print(all_data_name)

all_data_name = ['GO_PPI_DM-BP+CC+MF-threshold-3', 'GO_PPI_CE-BP+CC+MF-threshold-3', 'GO_PPI_SC-MF-threshold-3', 'GO_PPI_SC-CC-threshold-3', 'GO_PPI_DM-MF-threshold-3', 'GO_PPI_CE-CC-threshold-3', 'GO_PPI_CE-MF-threshold-3', 'GO_PPI_SC-BP+MF-threshold-3', 'GO_PPI_CE-CC+MF-threshold-3', 'GO_PPI_DM-CC+MF-threshold-3', 'GO_PPI_CE-BP+CC-threshold-3', 'GO_PPI_MM-BP+MF-threshold-3', 'GO_PPI_DM-BP+MF-threshold-3', 'GO_PPI_MM-CC-threshold-3', 'GO_PPI_MM-BP+CC-threshold-3', 'GO_PPI_DM-BP-threshold-3', 'GO_PPI_MM-CC+MF-threshold-3', 'GO_PPI_CE-BP+MF-threshold-3', 'GO_PPI_SC-BP+CC-threshold-3', 'GO_PPI_CE-BP-threshold-3', 'GO_PPI_SC-CC+MF-threshold-3', 'GO_PPI_DM-CC-threshold-3', 'GO_PPI_MM-BP-threshold-3', 'GO_PPI_SC-BP-threshold-3', 'GO_PPI_MM-MF-threshold-3', 'GO_PPI_SC-BP+CC+MF-threshold-3', 'GO_PPI_MM-BP+CC+MF-threshold-3', 'GO_PPI_DM-BP+CC-threshold-3']
visual_all("final_data", all_data_name)
# visualize(all_data)
# visualize(removed_train_data)
# visualize(added_train_data)

# df1 = pd.DataFrame({'a': [1, 2, 0], 'b': [5, 6, 5]})
# df2 = pd.DataFrame({'a': [0, 2], 'b': [5, 6]})
# print(get_diff(df1, df2))


