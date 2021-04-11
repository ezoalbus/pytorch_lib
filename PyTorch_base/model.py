import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """ ネットワークの定義 """
    def __init__(self):
        super(Net, self).__init__()
        # 28*28 dim -> 256 dim
        self.fc1 = nn.Linear(28*28, 256) 
        # 256 dim -> 10 dim
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # shapeを(batch size × dim_n)に変更
        # view()の１つ目の引数に-1を入れることで、２つ目の引数に合わせてshapeを調整
        x = x.view(-1, 28*28)
        # fc1に通す
        x = self.fc1(x)
        # activation func.を適用
        x = F.relu(x)
        # fc2に通す
        x = self.fc2(x)

        return x
