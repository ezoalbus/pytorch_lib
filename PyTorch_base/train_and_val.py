import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from model import Net
from data_loader import get_dataloader

from tqdm import tqdm


# gpu対応
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データローダーの定義
train_loader = get_dataloader(dataset='mnist', datatype='train')
test_loader = get_dataloader(dataset='mnist', datatype='test')

net = Net().to(device)

# 最適化の目的関数の定義
object_func = nn.CrossEntropyLoss()

# 最適化手法の選択
optimizer = optim.Adam(net.parameters(), lr=0.001)


#--- TRAIN ---#
epoch_n = 10
verbose = True
verbose_step = 100


for epoch in tqdm(range(epoch_n)):
    # 損失（表示用）
    running_loss = 0.0
    for i, (X_train, y_train) in enumerate(train_loader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        # 勾配初期化
        optimizer.zero_grad()
        # forward
        outputs = net(X_train)
        # 損失計算
        loss = object_func(outputs, y_train)
        # ↑ここまでで計算グラフの構築が行われ、勾配に使う情報が設定される
        # 勾配計算
        loss.backward()
        # パラメータ更新
        optimizer.step()
        # lossの更新
        running_loss += loss.item()

        if verbose and ((i + 1) % verbose_step == 0):
            print(f'epoch:{epoch + 1}, batch:{i + 1}, '
                + f'loss: {running_loss / verbose_step}')
            running_loss = 0.0


#--- TEST ---#

correct_n = 0
total_n = 0

# 予測時には勾配更新をしない
with torch.no_grad():
    for (X_test, y_test) in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = net(X_test)  # 更に、nn.functional.softmax() すると確率が得られる
        # torch.max() -> return (最大値, そのindex)
        _, predicted = torch.max(outputs.data, dim=1)
        total_n += y_test.size(0)
        correct_n += (predicted == y_test).sum().item()
print(f'Accuracy for test data: {100 * float(correct_n/total_n)}')
