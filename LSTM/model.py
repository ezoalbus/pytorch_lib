import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """ LSTMを用いた分類器のモデル"""
    # ネットワークの定義
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # 単語をベクトル化
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # LSTMの入力をバッチサイズ × 文章の長さ × ベクトル次元数にする(batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
        # 隠れ層(LSTM)の出力を受け取って全結合する
        self.hidden2target = nn.Linear(hidden_dim, target_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)  # dim=1 は行方向の指定
    
    # 順伝播
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # viewで3次元テンソルにする
        # テキスト分類をしたいので、２つ目の戻り値(最後の隠れ層の出力)を使用
        _, lstm_out = self.lstm(embeds)
        # lstm_out[0](３次元テンソル)を2次元にして全結合
        fc_out = self.hidden2target(lstm_out[0])
        # logsoftmaxを適用(squeeze()で、(batch_size × target_size) にする)
        predicted_score = self.logsoftmax(fc_out.squeeze())

        return predicted_score