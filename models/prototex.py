from functools import partialmethod

import numpy as np
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

import models.prototex


class PrototypeLayer(nn.Module):

    def __init__(self, num_features, num_prototypes):
        super().__init__()
        self.num_features = num_features
        self.num_prototypes = num_prototypes

        self.prototype_weights = nn.Parameter(torch.empty((num_prototypes, num_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.prototype_weights)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.unsqueeze(1)
        return torch.sqrt(torch.sum((inp - self.prototype_weights) ** 2, -1))


class Prototex(nn.Module):

    def __init__(self, params, num_classes=2):
        super().__init__()
        self.params = params
        self.bart_encoder_model = transformers.BartModel.from_pretrained('facebook/bart-base').encoder
        num_encoder_layers = len(self.bart_encoder_model.layers)
        for (i, layer) in enumerate(self.bart_encoder_model.layers):
            for param in layer.parameters():
                param.requires_grad = (i == num_encoder_layers - 1)

        encoder_out_dim = self.bart_encoder_model.config.d_model
        self.dropout = nn.Dropout(params['dropout'])

        self.encoder = nn.Sequential(
            nn.Linear(encoder_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, params['latent_dim'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(params['latent_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, encoder_out_dim)
        )

        self.proto_layer = PrototypeLayer(params['latent_dim'], params['num_prototypes'])

        self.out_layer = nn.Sequential(
            nn.Linear(params['num_prototypes'], num_classes),
        )

        self.num_classes = num_classes

    def predict(self, input_ids, attention_mask):
        eos_mask = input_ids.eq(int(self.bart_encoder_model.config.eos_token_id))

        last_hidden_state = self.bart_encoder_model(input_ids.to(self.device),
                                                    attention_mask.to(self.device)).last_hidden_state
        model_inp = last_hidden_state[eos_mask, :].view(last_hidden_state.size(0), -1, last_hidden_state.size(-1))[:,
                    -1, :]

        latent_state = self.encoder(model_inp)

        proto_out = self.proto_layer(latent_state)

        logits = self.out_layer(proto_out)
        return logits



    def forward(self, input_ids, attention_mask, y):
        eos_mask = input_ids.eq(int(self.bart_encoder_model.config.eos_token_id))

        last_hidden_state = self.bart_encoder_model(input_ids.to(self.device),
                                                    attention_mask.to(self.device)).last_hidden_state
        model_inp = last_hidden_state[eos_mask, :].view(last_hidden_state.size(0), -1, last_hidden_state.size(-1))[:,
                    -1, :]
        y = y.to(self.device)

        latent_state = self.encoder(model_inp)
        reconstruction = self.decoder(latent_state)

        proto_out = self.proto_layer(latent_state)

        logits = self.out_layer(proto_out)

        loss1 = torch.nn.CrossEntropyLoss()(logits, y)
        loss2 = torch.nn.MSELoss()(reconstruction, model_inp.detach())
        loss3 = torch.mean(torch.min(proto_out, 0)[0])
        loss4 = torch.mean(torch.min(proto_out, 1)[0])

        loss = loss1 + self.params['lambda'] * loss2 + self.params['lambda1'] * loss3 + self.params['lambda2'] * loss4

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

        latent_state = self.encoder(model_inp)

        proto_out = self.proto_layer(latent_state)

        logits = self.out_layer(proto_out)
        top = torch.topk(logits, 5)[1].view(-1)

        return [class_names[t] for t in top]

    @staticmethod
    def get_default_params():
        return {
            'lr': 3e-5,
            'weight_decay': 1e-3,
            'eps': 1e-8,
            'dropout': 0.2,
            'num_prototypes': 256,
            'lambda': 0.05,
            'lambda1': 0.05,
            'lambda2': 0.05,
            'latent_dim': 128,
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
            for batch in tqdm(loader, total=len(loader), desc=name):
                input_ids, attention_mask, y = batch
                a = self.predict(input_ids, attention_mask)
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

        model = models.prototex.Prototex(params, num_classes).to(device)
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
                torch.save(model.state_dict(), './checkpoints/model_{}.pt'.format(epoch))
