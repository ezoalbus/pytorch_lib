import torch
import torch.nn as nn
import re
import MeCab
import mojimoji

from prepare_data import prepare_data


class Text2IDseq:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")
        self.datasets = prepare_data()
        self.word2idx = self.make_wordidx_dic()
        self.vocab_size = len(self.word2idx)
        self.category_idx = self.make_category_idx()

    def wakati_MeCab(self, text):
        token_ls = self.tagger.parse(text).split()
        return token_ls

    def clean_text(self, text):
        # 記号削除
        text = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）\
                            ＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～\
                            ＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:\
                            ;<>?＜＞〔〕〈〉？、。・,\./『』【】「」\
                            →←○《》≪≫\n\u3000]+', "", text)
        # 全角 -> 半角
        # text = mojimoji.zen_to_han(text)
        return text

    def wakati_and_clean(self, text):
        token_ls = self.wakati_MeCab(text)
        token_ls = list(map(self.clean_text, token_ls))
        token_ls = [t for t in token_ls if t]
        return token_ls

    def make_wordidx_dic(self):
        """ 単語ID辞書の作成 """
        word2idx = {}
        # 系列の長さを揃えるため、<pad>を追加
        word2idx['<pad>'] = 0

        for text in self.datasets['title']:
            token_ls = self.wakati_and_clean(text)
            for token in token_ls:
                if token in word2idx:
                    continue
                word2idx[token] = len(word2idx)
        return word2idx

    def make_dataset_idx(self):
        datasets_titleIdx_buf = []
        datasets_categoryIdx = []
        for title, category in zip(self.datasets['title'], self.datasets['category']):
            title_idx = self.convert_text2idseq(title)
            category_idx = self.get_category_idx(category)
            datasets_titleIdx_buf.append(title_idx)
            datasets_categoryIdx.append(category_idx)

        datasets_titleIdx = []
        # 系列の最大長
        max_title_len = max(map(len, datasets_titleIdx_buf))
        for title in datasets_titleIdx_buf:
            # padding
            pad_ls = torch.tensor([0] * (max_title_len - len(title)))
            title = torch.cat([pad_ls, title])
            datasets_titleIdx.append(title)

        datasets_categoryIdx = torch.tensor(datasets_categoryIdx, dtype=torch.long)

        return datasets_titleIdx, datasets_categoryIdx


    def convert_text2idseq(self, text):
        token_ls = self.wakati_and_clean(text)
        idx_tensor = torch.tensor([self.word2idx[t] for t in token_ls])
        return idx_tensor

    def make_sents_matrix(self, text, embedding_dim):
        idx_tensor = self.convert_text2idseq(text)
        embeds = nn.Embedding(self.vocab_size, embedding_dim)
        sents_matrix = embeds(idx_tensor)
        return sents_matrix


    def get_sents_matrix(self, text='私は猫が好きです', embedding_dim=10):
        sents_matrix = self.make_sents_matrix(text, embedding_dim)
        print(sents_matrix)

        # バッチサイズを1にする
        sents_matrix = sents_matrix.view(len(sents_matrix), 1, -1)
        print(sents_matrix.size())
        print(sents_matrix)
        return sents_matrix

    def make_category_idx(self):
        categories = self.datasets['category']
        category_idx = {}
        for c in categories:
            if c in category_idx: 
                continue
            category_idx[c] = len(category_idx)
        return category_idx

    def get_category_idx(self, category):
        category = self.category_idx[category]
        
        return category

if __name__ == '__main__':
    text2idseq = Text2IDseq()
    text, cat = text2idseq.make_dataset_idx()
    # print(text)
    print(cat)