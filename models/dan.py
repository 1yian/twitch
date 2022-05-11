import torch
import torch.nn as nn
import numpy as np
import nltk.tokenize

import models.utils as utils


class DeepAveragingNetwork(nn.Module):
    def __init__(self, params, vocab_and_freq_dict):
        super().__init__()
        self.num_labels = params['num_labels']
        self.embedding_size = params['embedding_size']

        self.tokenizer = nltk.tokenize.TweetTokenizer()

        self.indexer = utils.WordIndexer(vocab_and_freq_dict, min_frequency=10)

        self.embedding_layer = nn.Embedding(len(self.indexer), self.embedding_size, padding_idx=self.indexer.padding_val)

        self.classfn_model = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(256, self.num_labels),
        )

    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        x = self.embedding_layer(input_tensor)
        x = torch.mean(x, 1)
        x = self.classfn_model(x)
        return x

    def transform_sentences_to_input(self, sentence_list):
        indices_list = []
        for word_list in sentence_list:
            single_index_list = []
            for word in word_list:
                single_index_list.append(self.indexer.index_of(word))
            indices_list.append(torch.LongTensor(single_index_list))
            indices_batch = torch.nn.utils.rnn.pad_sequence(indices_list, batch_first=True,
                                                            padding_value=self.indexer.PADDING_VALUE)
        return indices_batch

    def train_model(self, dataloader, weight):
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        optim = torch.optim.Adam(self.parameters())
        total_losses = []
        for batch in dataloader:
            optim.zero_grad()
            sentences, labels = list(zip(*batch))[0], list(zip(*batch))[1]
            labels = torch.stack(labels)
            labels = torch.nn.functional.normalize(labels, p=1, dim=1).to(self.device)
            model_input = self.transform_sentences_to_input(sentences).to(self.device)
            pred_logits = self(model_input)
            pred_log_dist = nn.functional.log_softmax(pred_logits, dim=1)
            loss = loss_fn(pred_log_dist, labels)
            loss.backward()
            optim.step()
            total_losses.append(loss.detach().clone().cpu().numpy())
        return np.mean(total_losses)

    def predict(self, text):
        word_list = nltk.tokenize.TweetTokenizer().tokenize(text)
        index_list = []
        for word in word_list:
            index_list.append(self.indexer.index_of(word))
        inp = torch.LongTensor(index_list).to(self.device).unsqueeze(0)
        pred_logits = self(inp)
        indices = torch.topk(pred_logits, 3, dim=1).indices

        return indices


    @staticmethod
    def get_default_params():
        return {
            'dropout': 0.2,
            'num_labels': 294,
            'embedding_size': 128,
        }

    @property
    def device(self):
        return next(self.parameters()).device
