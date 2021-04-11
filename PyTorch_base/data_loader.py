import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloader(dataset='mnist', datatype='train', datadir='./data'):
    # データ変換
    transform = transforms.Compose(
        [
            # ndarrayテンソルに変換
            transforms.ToTensor(),
            # mean0.5、std0.5に正規化。チャンネル数のタプルで渡す。
            transforms.Normalize(mean=(0.5, ), std=(0.5, ))
        ]
    )

    if dataset == 'mnist':
        if datatype == 'train':
            # 学習データの準備
            train_data = torchvision.datasets.MNIST(
                root=datadir,  # 保存先
                train=True,
                download=True,
                transform=transform
            )

            # 学習用ローダーの定義
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=128,
                shuffle=True,
                num_workers=2
            )

            return train_loader

        if datatype == 'test':
            # テストデータの準備
            test_data = torchvision.datasets.MNIST(
                root=datadir,
                train=False,
                download=True,
                transform=transform
            )

            # テスト用ローダーの定義
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=128,
                shuffle=False,
                num_workers=2
            )

            return test_loader

