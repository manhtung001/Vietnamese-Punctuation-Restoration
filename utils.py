from underthesea import text_normalize, word_tokenize
import re
import torch
import torch.nn as nn
import argparse
from transformers import *

phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

TOKEN_IDX = {
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    }
}

# 'O' -> No punctuation
punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}

# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'vinai/phobert-base': (phobert, tokenizer, 768, 'roberta')
}

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}


# Normalize txt
def textNorm(txt):
    return text_normalize(txt)


def norm4PhoBert(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


def preprocessing(txt, dictmap=dict_map):
    # txt norm (underthesea)
    txt = textNorm(txt)

    # norm mark 4 phobert
    txt = norm4PhoBert(txt, dictmap)

    # tokenizer
    txt_tokens = word_tokenize(txt)

    # replace space w _
    txt_tokens = [i.replace(' ', '_') for i in txt_tokens]

    return ' '.join(txt_tokens)


class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)
        self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.GRU(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)  ##### GRU
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


class DeepPunctuationCRF(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuationCRF, self).__init__()
        self.bert_lstm = DeepPunctuation(pretrained_model, freeze_bert, lstm_dim)
        self.crf = CRF(len(punctuation_dict), batch_first=True)

    def log_likelihood(self, x, attn_masks, y):
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        dec_out = self.crf.decode(x, mask=attn_masks)
        y_pred = torch.zeros(y.shape).long().to(y.device)
        for i in range(len(dec_out)):
            y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        return y_pred


def parse_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--pretrained-model', default='vinai/phobert-base', type=str, help='pretrained language model')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')
    parser.add_argument('--language', default='vietnamse', type=str, help='language English (en) oe Bangla (bn)')
    parser.add_argument('--in-file', default='tmp/inference', type=str, help='path to inference file')
    parser.add_argument('--weight-path', default='phobert-base-gru_weights.pt', type=str,
                        help='model weight path')  ### MODEL PATH HERE, WE ARE USING PHOBERT-BASE-GRU
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--out-file', default='tmp/inference_out.txt', type=str, help='output file location')

    args = parser.parse_args([])
    return args


args = parse_arguments()

# tokenizer
tokenizer = MODELS[args.pretrained_model][1]
token_style = MODELS[args.pretrained_model][3]

# logs
model_save_path = args.weight_path

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
print("device:", device)
if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)


def inference(txt):
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    deep_punctuation.eval()
    text = txt
    text = re.sub(r"[,:\-–.!;?]", '', text)
    words_original_case = text.split()
    words = text.lower().split()

    word_pos = 0
    sequence_len = args.sequence_length
    result = ""
    decode_idx = 0
    punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
    if args.language != 'en':
        punctuation_map[2] = '।'

    while word_pos < len(words):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y_mask = [0]

        while len(x) < sequence_len and word_pos < len(words):
            tokens = tokenizer.tokenize(words[word_pos])
            if len(tokens) + len(x) >= sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                    y_mask.append(0)
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                y_mask.append(1)
                word_pos += 1
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y_mask.append(0)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

        x = torch.tensor(x).reshape(1, -1)
        y_mask = torch.tensor(y_mask)
        attn_mask = torch.tensor(attn_mask).reshape(1, -1)
        x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

        with torch.no_grad():
            if args.use_crf:
                y = torch.zeros(x.shape[0])
                y_predict = deep_punctuation(x, attn_mask, y)
                y_predict = y_predict.view(-1)
            else:
                y_predict = deep_punctuation(x, attn_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1
    print('Punctuated text')
    print(result)
    # with open(args.out_file, 'w', encoding='utf-8') as f:
    #     f.write(result)
    return result
