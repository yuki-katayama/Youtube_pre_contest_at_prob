import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import umap
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

train = pd.read_csv("../input/youtube/train_data.csv")
train_true = pd.read_csv("../input/youtube/train_data.csv")
test_data = pd.read_csv("../input/youtubetest/test_data.csv")

# 最後のデータ結合の為idは別で保存しておく
iddf = test_data[["id"]]

train.head()

# 目的変数を可視化する
plt.hist(train['y'], bins=20, log=True)

# 訓練データとテストデータを別々で処理するのは面倒なので、結合して前処理していく
data = pd.concat([train, test_data])

# あとでテストと訓練を分別できる様に訓練データの長さを保存
train_len = len(train["y"])
print(train.shape, test_data.shape, data.shape)

# 相関係数を見て、関係のありそうな説明変数を見る
corrmat = data.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=.8, square=True)
# 下の相関係数の図からyとはlikes,dislikes,comment_countがかろうじて高いと分かる

# 各IDに所属しているyの「平均値、最大値、最小値、標準偏差、個数」
mean_ = train[["categoryId", "y"]].groupby(
    "categoryId").mean().reset_index().rename({"y": "mean"}, axis=1)
max_ = train[["categoryId", "y"]].groupby(
    "categoryId").max().reset_index().rename({"y": "max"}, axis=1)
min_ = train[["categoryId", "y"]].groupby(
    "categoryId").min().reset_index().rename({"y": "min"}, axis=1)
std_ = train[["categoryId", "y"]].groupby(
    "categoryId").std().reset_index().rename({"y": "std"}, axis=1)
count_ = train[["categoryId", "y"]].groupby(
    "categoryId").count().reset_index().rename({"y": "count"}, axis=1)


# 各IDの部位点
q1_ = train[["categoryId", "y"]].groupby("categoryId").quantile(
    0.1).reset_index().rename({"y": "q1"}, axis=1)
q25_ = train[["categoryId", "y"]].groupby("categoryId").quantile(
    0.25).reset_index().rename({"y": "q25"}, axis=1)
q5_ = train[["categoryId", "y"]].groupby("categoryId").quantile(
    0.5).reset_index().rename({"y": "q5"}, axis=1)
q75_ = train[["categoryId", "y"]].groupby("categoryId").quantile(
    0.75).reset_index().rename({"y": "q75"}, axis=1)
q9_ = train[["categoryId", "y"]].groupby("categoryId").quantile(
    0.9).reset_index().rename({"y": "q9"}, axis=1)


# 目的変数を対数変換して、正則化してからドロップする
train["y"] = np.log1p(train["y"])
y = train["y"]
del train["y"]

# 型をbool方からintに変換
data["comments_disabled"] = data["comments_disabled"].astype(np.int16)
data["ratings_disabled"] = data["ratings_disabled"].astype(np.int16)
# False = 0: True = 1

# like dislike comment
#data["likes2"] = data["likes"]**2
data["loglikes"] = np.log(data["likes"]+1)
#data["dislikes2"] = data["dislikes"]**2
data["logdislikes"] = np.log(data["dislikes"]+1)
data["logcomment_count"] = np.log(data["comment_count"]+1)

data["sqrtlikes"] = np.sqrt(data["likes"])

# ここの+1は何かの微調整わからん。。
data["like_dislike_ratio"] = data["likes"]/(data["dislikes"]+1)
data["comments_like_ratio"] = data["comment_count"]/(data["likes"]+1)
data["comments_dislike_ratio"] = data["comment_count"]/(data["dislikes"]+1)

# likes comments diable
data["likes_com"] = data["likes"] * data["comments_disabled"]
data["dislikes_com"] = data["dislikes"] * data["comments_disabled"]
data["comments_likes"] = data["comment_count"] * data["ratings_disabled"]

# tag
# 欠損値を処理
data["tags"].fillna("[none]", inplace=True)

# "|"を含んでいる文で、"|"で区切って、カウントして、ソート
# それぞれのtagの出現回数を表している
tagdic = dict(pd.Series("|".join(list(data["tags"])).split(
    "|")).value_counts().sort_values())

# tagの個数をカウント
data["num_tags"] = data["tags"].astype(str).apply(lambda x: len(x.split("|")))

# 文字数
data["length_tags"] = data["tags"].astype(str).apply(lambda x: len(x))

# 使っているtagをtagdicでポイント化して、マイナーか、有名なtagを使っているかを判定できる
data["tags_point"] = data["tags"].apply(
    lambda tags: sum([tagdic[tag] for tag in tags.split("|")]))

data.head()

# 日付型に変換
data["publishedAt"] = pd.to_datetime(data["publishedAt"], utc=True)

data["publishedAt_year"] = data["publishedAt"].apply(lambda x: x.year)
data["publishedAt_month"] = data["publishedAt"].apply(lambda x: x.month)
data["publishedAt_day"] = data["publishedAt"].apply(lambda x: x.day)
data["publishedAt_hour"] = data["publishedAt"].apply(lambda x: x.hour)
data["publishedAt_minute"] = data["publishedAt"].apply(lambda x: x.minute)
#df["publishedAt_second"] = df["publishedAt"].apply(lambda x: x.second)
data["publishedAt_dayofweek"] = data["publishedAt"].apply(
    lambda x: x.dayofweek)

#df["collection_date_year"] = df["collection_date"].apply(lambda x: int(x[0:2]))
data["collection_date_month"] = data["collection_date"].apply(
    lambda x: int(x[3:5]))
data["collection_date_day"] = data["collection_date"].apply(
    lambda x: int(x[6:8]))

# "collection_date"をdatetime型に変換
data["collection_date"] = pd.to_datetime(
    "20"+data["collection_date"], format="%Y.%d.%m", utc=True)
data.head()

# delta
# 公開日からデータ収集日までの期間
data["delta"] = (data["collection_date"] -
                 data["publishedAt"]).apply(lambda x: x.days)

data["logdelta"] = np.log(data["delta"])
data["sqrtdelta"] = np.sqrt(data["delta"])


data["published_delta"] = (data["publishedAt"] -
                           data["publishedAt"].min()).apply(lambda x: x.days)
data["collection_delta"] = (
    data["collection_date"] - data["collection_date"].min()).apply(lambda x: x.days)

# 欠損値は空白は埋める
data["description"].fillna(" ", inplace=True)

# 全てを小文字に変換して、httpをカウント
data["ishttp_in_dis"] = data["description"].apply(
    lambda x: x.lower().count("http"))

# descriptionの長さ
data["len_description"] = data["description"].apply(lambda x: len(x))

data["title"].fillna(" ", inplace=True)
# 欠損値がある場合空白で埋める

data["len_title"] = data["title"].apply(lambda x: len(x))
# タイトルの長さ


# 文字列stringに1文字でも「ひらがな」「カタカナ」「漢字」のどれかが含まれていればTrueを返します

def is_japanese(string):
    for ch in string:
        try:
            name = unicodedata.name(ch)
            if "CJK UNIFIED" in name \
                    or "HIRAGANA" in name \
                    or "KATAKANA" in name:
                return True
        except:
            continue
    return False


# is japanese
data["isJa_title"] = data["title"].apply(lambda x: is_japanese(x))
data["isJa_tags"] = data["tags"].apply(lambda x: is_japanese(x))
data["isJa_description"] = data["description"].apply(lambda x: is_japanese(x))

# is englosh
# 全てが英数字のみか判定する
# 文字コードを指定しないとうまく判定できない
data["onEn_title"] = data["title"].apply(lambda x: x.encode('utf-8').isalnum())
data["onEn_tags"] = data["tags"].apply(lambda x: x.encode('utf-8').isalnum())
data["onEn_description"] = data["description"].apply(
    lambda x: x.encode('utf-8').isalnum())


# cotain englosh
# findallはマッチするすべての部分文字列をリストにして返す
# またlenを使ってリストの長さを図ることで、マッチした文字の長さだけを見ることができる
# x.lowerを使う理由は処理が多少早くなるから！  (使わなくても結果は変わらない)
data["conEn_title"] = data["title"].apply(
    lambda x: len(re.findall(r'[a-zA-Z0-9]', x.lower())))
data["conEn_tags"] = data["tags"].apply(
    lambda x: len(re.findall(r'[a-zA-Z0-9]', x.lower())))
data["conEn_description"] = data["description"].apply(
    lambda x: len(re.findall(r'[a-zA-Z0-9]', x.lower())))

data.head()

# Music
data["music_title"] = data["title"].apply(lambda x: "music" in x.lower())
data["music_tags"] = data["tags"].apply(lambda x: "music" in x.lower())
data["music_description"] = data["description"].apply(
    lambda x: "music" in x.lower())

# Official
data["isOff"] = data["title"].apply(lambda x: "fficial" in x.lower())
data["isOffChannell"] = data["channelTitle"].apply(
    lambda x: "fficial" in x.lower())
data["isOffJa"] = data["title"].apply(lambda x: "公式" in x.lower())
data["isOffChannellJa"] = data["channelTitle"].apply(
    lambda x: "公式" in x.lower())

# 公式アカウントを知るために、"公式"と"fficialの2パターンを考える
# fficalの理由は Oが大文字と小文字、空白がある場合があるから

# CMの場合の処理
data["cm_title"] = data["title"].apply(lambda x: "cm" in x.lower())
data["cm_tags"] = data["tags"].apply(lambda x: "cm" in x.lower())
data["cm_description"] = data["description"].apply(lambda x: "cm" in x.lower())

# 最初に求めた奴の結合
data = data.merge(mean_, how='left', on=["categoryId"])
data = data.merge(max_, how='left', on=["categoryId"])
data = data.merge(min_, how='left', on=["categoryId"])
data = data.merge(std_, how='left', on=["categoryId"])
#data = df.merge(count_, how='left', on=["categoryId"])
data = data.merge(q1_, how='left', on=["categoryId"])
data = data.merge(q25_, how='left', on=["categoryId"])
data = data.merge(q5_, how='left', on=["categoryId"])
data = data.merge(q75_, how='left', on=["categoryId"])
data = data.merge(q9_, how='left', on=["categoryId"])

# 出現頻度

for col in ["categoryId", "channelTitle"]:
    freq = data[col].value_counts()
    # map(freq)で{1:10,2:30}とかになっている
    data["freq_"+col] = data[col].map(freq)


# 表示される列数と、行数を増やした
# 削除する列の確認が不便だったので追加した
pd.set_option('display.max_columns', 68)
pd.set_option('display.max_rows', 100)

del data["id"]
del data["video_id"]
del data["title"]
del data["publishedAt"]
del data["channelId"]
del data["channelTitle"]
del data["collection_date"]
del data["tags"]
del data["thumbnail_link"]
del data["description"]
del data["y"]

# ここの欠損地は全てテストデータの物なので偏らない様に最近傍法用のデータを作る
data_kmeans = data
data_kmeans["mean"] = data["mean"].fillna(np.mean(data["mean"]))
data_kmeans["max"] = data["max"].fillna(np.mean(data["max"]))
data_kmeans["min"] = data["min"].fillna(np.mean(data["min"]))
data_kmeans["std"] = data["std"].fillna(np.mean(data["std"]))
data_kmeans["q1"] = data["q1"].fillna(np.mean(data["q1"]))
data_kmeans["q25"] = data["q25"].fillna(np.mean(data["q25"]))
data_kmeans["q5"] = data["q5"].fillna(np.mean(data["q5"]))
data_kmeans["q75"] = data["q75"].fillna(np.mean(data["q75"]))
data_kmeans["q9"] = data["q9"].fillna(np.mean(data["q9"]))


# スケール変換
scalar = StandardScaler()
scalar.fit(data.astype(np.float32))
data = pd.DataFrame(scalar.transform(data), columns=data.columns)
data_kmeans = pd.DataFrame(scalar.transform(data_kmeans), columns=data.columns)

kmeans = KMeans(n_clusters=100, random_state=0)
clusters = kmeans.fit(data_kmeans)
data["cluster"] = clusters.labels_
print(data["cluster"].unique())
data.head()

# PCA
X = data
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df.reset_index(drop=True)
pca_df["cluster"] = data["cluster"].reset_index(drop=True)

for i in pca_df["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"] == i]
    plt.scatter(tmp[0], tmp[1])

# UMAP
um = umap.UMAP()
um.fit(data)
data_1 = um.transform(data)
plt.scatter(data_1[:, 0], data_1[:, 1], c=clusters.labels_, cmap=cm.tab10)
plt.colorbar()

# 訓連用と評価用に分ける
X = data.iloc[:train_len, :]
test = data.iloc[train_len:, :]
print(X.shape, y.shape, test.shape)

# 訓練用とテスト用に分ける
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, shuffle=True)

print(X_train.shape, y_train.shape)


# RMLSE を計算する

def rmsle_(y_valid, y_pred):
    rmsle = np.sqrt(mean_squared_log_error(np.exp(y_valid), np.exp(y_pred)))
    return rmsle


parameters = {
    'n_estimators': [3, 5, 10, 30, 50],
    'random_state': [7, 42],
    'max_depth': [3, 5, 8, 10],
}

# 交差検証のパラメータ
kfold_cv = KFold(n_splits=5, shuffle=True)

# グリッドサーチで良いパラメータを探す
Grid = GridSearchCV(estimator=RandomForestRegressor(),
                    param_grid=parameters, cv=3)
Grid.fit(X_train, y_train)

best_model = Grid.best_estimator_

train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

print("訓練での認識精度:" + str(train_score))
print("テストデータでの認識精度   :" + str(test_score))

print(Grid.best_estimator_)

# モデルの評価

y_pred_test_RF = best_model.predict(X_test)

# 視聴率は0未満にならない
y_pred_test_RF[y_pred_test_RF < 0] = 0
RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred_test_RF))

print(RMSLE)

a = test.isnull().sum()
a = pd.DataFrame(a)

test["mean"] = test["mean"].fillna(np.mean(test["mean"]))
test["max"] = test["max"].fillna(np.mean(test["max"]))
test["min"] = test["min"].fillna(np.mean(test["min"]))
test["std"] = test["std"].fillna(np.mean(test["std"]))
test["q1"] = test["q1"].fillna(np.mean(test["q1"]))
test["q25"] = test["q25"].fillna(np.mean(test["q25"]))
test["q5"] = test["q5"].fillna(np.mean(test["q5"]))
test["q75"] = test["q75"].fillna(np.mean(test["q75"]))
test["q9"] = test["q9"].fillna(np.mean(test["q9"]))

test.isnull().sum()

y_pred_eval_RF = best_model.predict(test)
y_pred_eval_RF = np.exp(y_pred_eval_RF)

# 対数変換していたyを戻す
df_sub_pred = pd.DataFrame(y_pred_eval_RF).rename(columns={0: "y"})
df_sub_pred = pd.concat([test_data['id'], df_sub_pred['y']], axis=1)
df_sub_pred.to_csv("randomforest.csv", index=False)


def rmsle(preds, data):
    y_true = data.get_label()
    y_pred = preds
    y_pred[y_pred < 0] = 0
    y_true[y_true < 0] = 0
    acc = np.sqrt(mean_squared_log_error(np.exp(y_true), np.exp(y_pred)))
    # name, result, is_higher_better
    return 'accuracy', acc, False


# Optunaの最適化パラメータを代入する
light_params = {'task': 'train',  # 他にpredict、convert_model、refitなどもある
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',  # 評価関数
                'verbosity': -1,  # 学習途中を表示しない,-1で表示しない
                "seed": 42,  # シード値
                'learning_rate': 0.01,  # 学習率
                'feature_fraction': 0.7,  # 学習の高速化と過学習の抑制に使用
                'num_leaves': 210}  # ノードの数→葉が多いほど複雑になる

best_params = {'lambda_l1': 0.019918875912078603, 'lambda_l2': 0.002616688073257713, 'num_leaves': 219,
               'feature_fraction': 0.6641013611124621, 'bagging_fraction': 0.7024199018549259, 'bagging_freq': 5, 'min_child_samples': 5}
#best_params =  {}
light_params.update(best_params)

xgb_params = {'learning_rate': 0.1,
              'objective': 'reg:squarederror',
              'eval_metric': 'rmse',
              'seed': 42,
              'tree_method': 'hist'}
best_params = {'learning_rate': 0.01665914389764044,
               'lambda_l1': 4.406831762257336, 'num_leaves': 39}
#best_params = {}
xgb_params.update(best_params)


FOLD_NUM = 11
kf = KFold(n_splits=FOLD_NUM,
           shuffle=True,
           random_state=42)

scores = []
feature_importance_df = pd.DataFrame()
pred_cv = np.zeros(len(test.index))
num_round = 10000


# iは交差検証の回数 tdxとvdxはランダムに選ばれたtrain,testのインデックス
for i, (tdx, vdx) in enumerate(kf.split(X, y)):
    print(f'Fold : {i}')
    # LGB
    # なぜy.valuesなのか　これだけわかればここのコード行ける！
    X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.values[tdx], y.values[vdx]

    # LGB
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    gbc = lgb.train(light_params, lgb_train, num_boost_round=num_round,
                    valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                    # feval=rmsle,
                    early_stopping_rounds=100, verbose_eval=500)
    if i == 0:
        importance_df = pd.DataFrame(
            gbc.feature_importance(), index=X.columns, columns=['importance'])
    else:
        importance_df += pd.DataFrame(gbc.feature_importance(),
                                      index=X.columns, columns=['importance'])
    gbc_va_pred = np.exp(gbc.predict(
        X_valid, num_iteration=gbc.best_iteration))
    gbc_va_pred[gbc_va_pred < 0] = 0

    # XGB
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(xgb_params, xgb_dataset, 10000, evals=[(xgb_dataset, 'train'), (xgb_test_dataset, 'eval')],
                     early_stopping_rounds=100, verbose_eval=500)
    xgbm_va_pred = np.exp(xgbm.predict(xgb.DMatrix(X_valid)))
    xgbm_va_pred[xgbm_va_pred < 0] = 0

    # ENS
    # lists for keep results
    lgb_xgb_rmsle = []
    lgb_xgb_alphas = []

    for alpha in np.linspace(0, 1, 101):  # (0 ~ 1までランダムな数を100回生成する)
        # このlgbとxgbアンサンブルめちゃくちゃ旨い！
        y_pred = alpha*gbc_va_pred + (1 - alpha)*xgbm_va_pred
        rmsle_score = np.sqrt(mean_squared_log_error(np.exp(y_valid), y_pred))

        # スコアとランダムな数値を格納
        lgb_xgb_rmsle.append(rmsle_score)
        lgb_xgb_alphas.append(alpha)

    # ndarrayに変換
    lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
    lgb_xgb_alphas = np.array(lgb_xgb_alphas)

    # rmsleが一番低い(ベストなスコア)時のインデックスを得て、alphaを格納
    lgb_xgb_best_alpha = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]

    print('best_rmsle=', lgb_xgb_rmsle.min())
    print('best_alpha=', lgb_xgb_best_alpha)

    plt.plot(lgb_xgb_alphas, lgb_xgb_rmsle)
    plt.title('f1_score for ensemble')
    plt.xlabel('alpha')
    plt.ylabel('f1_score')
    plt.legend()

    score_ = lgb_xgb_rmsle.min()
    scores.append(score_)
    # 一番良かったスコアをappendする

    lgb_submission = np.exp(gbc.predict(
        (test), num_iteration=gbc.best_iteration))
    lgb_submission[lgb_submission < 0] = 0

    xgbm_submission = np.exp(xgbm.predict(xgb.DMatrix(test)))
    xgbm_submission[xgbm_submission < 0] = 0

    submission = lgb_xgb_best_alpha*lgb_submission + \
        (1 - lgb_xgb_best_alpha)*xgbm_submission  # 上のアンサンブルの仕方と一緒

    # FOLD_NUM個分の答えになっているので割る
    # これをFOLD_NUM回繰り返すのでpred_cvは良い感じの答えになる
    pred_cv += submission/FOLD_NUM

print("##########")
print(np.mean(scores))

light_submission_df = pd.concat([iddf, pd.DataFrame(pred_cv)], axis=1)
light_submission_df.columns = ["id", "y"]
light_submission_df.to_csv("submission_lgb.csv", index=False)
print("end")
