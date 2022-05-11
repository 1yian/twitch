from functools import partialmethod

import numpy as np
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

import models.bart_classifier


class BartClassifier(nn.Module):

    def __init__(self, params, num_classes=2):
        super().__init__()
        self.bart_encoder_model = transformers.BartModel.from_pretrained('facebook/bart-base').encoder
        num_encoder_layers = len(self.bart_encoder_model.layers)

        for (i, layer) in enumerate(self.bart_encoder_model.layers):
            for param in layer.parameters():
                param.requires_grad = (i == num_encoder_layers - 1)

        encoder_out_dim = self.bart_encoder_model.config.d_model
        self.dropout = nn.Dropout(params['dropout'])
        self.output_layer = nn.Linear(encoder_out_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, y):
        eos_mask = input_ids.eq(int(self.bart_encoder_model.config.eos_token_id))

        last_hidden_state = self.bart_encoder_model(input_ids.to(self.device),
                                                    attention_mask.to(self.device)).last_hidden_state
        model_inp = last_hidden_state[eos_mask, :].view(last_hidden_state.size(0), -1, last_hidden_state.size(-1))[:,
                    -1, :]

        logits = self.output_layer(model_inp)
        loss = self.loss_fn(logits, y.to(self.device))

        return logits, loss

    def predict_on_string(self, string, class_names):

        encoding = self.get_tokenizer()(string.split(), padding='max_length',
                             truncation=True, max_length=256, return_attention_mask=True)
        inp_id, attn_mask = torch.LongTensor(encoding['input_ids'][0]).unsqueeze(0), \
                            torch.LongTensor(encoding['attention_mask'][0]).unsqueeze(0)

        eos_mask = inp_id.eq(int(self.bart_encoder_model.config.eos_token_id))

        last_hidden_state = self.bart_encoder_model(inp_id.to(self.device),
                                                    attn_mask.to(self.device)).last_hidden_state
        model_inp = last_hidden_state[eos_mask, :].view(last_hidden_state.size(0), -1, last_hidden_state.size(-1))[:,
                    -1, :]

        logits = self.output_layer(model_inp)[0]
        top = torch.topk(logits, 5)[1]
        print(top.shape)
        return [class_names[t] for t in top]

    @staticmethod
    def get_default_params():
        return {
            'lr': 3e-5,
            'weight_decay': 1e-3,
            'eps': 1e-8,
            'dropout': 0.2,
        }

    @staticmethod
    def get_hyperparam_space():
        return {
            'lr': hp.loguniform('lr', 1e-6, 1e-2),
        }

    @staticmethod
    def get_tokenizer():
        return transformers.BartTokenizerFast.from_pretrained('facebook/bart-base', add_prefix_space=True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def tokenizer(self):
        return type(self).get_tokenizer()

    def predict_on_loader(self, loader, name):
        y_scores = []
        y_true = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=name):
                input_ids, attention_mask, y = batch
                a, loss = self(input_ids, attention_mask, y)
                y_scores.append(torch.softmax(a, 1))
                y_true.append(y)
            ret_y_scores = torch.cat(y_scores, 0).detach().cpu().numpy()
            ret_y_true = torch.cat(y_true, 0).detach().cpu().numpy()

        return top_k_accuracy_score(ret_y_true, ret_y_scores, k=5, labels=range(self.num_classes))


    @staticmethod
    def train_model(train_dl, val_dl, test_dl, train_eval_dl, num_classes, print_output=True, params=None,
                    class_names=None):
        if not print_output:
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        params = params if params is not None else models.bart_classifier.BartClassifier.get_default_params()

        model = BartClassifier(params, num_classes).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
                                  eps=params['eps'])

        max_num_iters = 1_000
        last_acc = "N/A"
        for epoch in range(max_num_iters):
            model.train()
            total_loss = []
            pbar = tqdm(train_dl, total=len(train_dl), unit='batches', desc='training')
            for batch in pbar:

                optim.zero_grad()
                input_ids, attention_mask, y = batch
                a, loss = model(input_ids, attention_mask, y)
                loss.backward()
                optim.step()
                total_loss.append(loss.detach().cpu().numpy())
                pbar.set_postfix({"loss": "{}".format(np.mean(total_loss)), "top_k_acc": last_acc})

            if epoch % 5 == 1:
                last_acc = model.predict_on_loader(val_dl, name='evaluating_val'), \
                           model.predict_on_loader(test_dl, name='evaluating_test')
                print(model.predict_on_string("Hewwo my queen", class_names))
                print(model.predict_on_string("wtf that was hype", class_names))
                print(model.predict_on_string("NA ULT", class_names))
                torch.save(model.state_dict(), './checkpoints_bart/model_{}.pt'.format(epoch))


