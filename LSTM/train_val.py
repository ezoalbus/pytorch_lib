import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from prepare_data import prepare_data
from preprocess import Text2IDseq
from model import LSTMClassifier
from data_loader import get_dataloader

# gpu対応
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text2idseq = Text2IDseq()

# 単語のベクトル次元数
EMBEDDING_DIM = 200
# 隠れ層の次元数
HIDDEN_DIM = 128
# データ全体の単語数
VOCAB_SIZE = text2idseq.vocab_size
# 分類先のカテゴリの数
TAG_SIZE = len(text2idseq.category_idx)
# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
model = model.to(device)
# 損失関数はNLLLoss()を使う。LogSoftmaxを使う時はこれを使うらしい。
loss_function = nn.NLLLoss()
# 最適化の手法はAdamで。lossの減りに時間かかるけど、一旦はこれを使う。
optimizer = optim.Adam(model.parameters(), lr=0.01)

dataloader = get_dataloader(batch_size=100)

# 各エポックの合計loss値を格納する
losses = []
# 50ループ回してみる。（バッチ化とかGPU使ってないので結構時間かかる...）
for epoch in range(1):
    all_loss = 0
    for title, category in dataloader['train']:
        loss = 0

        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()

        # 文章を単語IDの系列に変換（modelに食わせられる形に変換）
        # print(text2idseq.word2idx)
        title = title.to(device)
        # 順伝播の結果を受け取る
        out = model(title)
        print(out.shape)
        # print(out)
        # 正解カテゴリをテンソル化
        # category = cat.long()
        category = category.to(device)
        category = category.flatten()
        print(category.shape)
        # print(category)
        # print(out)
        # 正解とのlossを計算
        loss = loss_function(out, category)
        # 勾配をセット
        loss.backward()
        # 逆伝播でパラメータ更新
        optimizer.step()
        # lossを集計
        all_loss += loss.item()
    losses.append(all_loss)
    print("epoch", epoch, "\t" , "loss", all_loss)
print("done.")

# テストデータの母数計算
test_num = len(dataloader['test'])
# 正解の件数
a = 0
# 勾配自動計算OFF
with torch.no_grad():
    for title, category in dataloader['test']:
        # テストデータの予測
        title = title.to(device)
        # category = category.long()
        category = category.to(device)
        category = category.flatten()
        out = model(title)
        # out = out.flatten()
        print(out.dtype)
        # outの一番大きい要素を予測結果をする
        _, predicts = torch.max(out, 1)
        print(predicts, category)

print("predict : ", a / test_num)
# predict :  0.6118391323994578