from tqdm import tqdm
import torch
import numpy
from sklearn.metrics import precision_recall_fscore_support


def evaluate_propaganda(dataloader, model, num_classes, tqdm_name=""):
    loader = tqdm(dataloader, total=len(dataloader), unit='batches', desc=tqdm_name)

    previously_training = model.training
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_len = 0

        y_pred, y_true = [], []
        if len(dataloader) > 0:
            for batch in loader:
                input_ids, attention_mask, y = batch

                y[y > num_classes - 1] = num_classes - 1
                pred_logits, loss, *_ = model(input_ids, attention_mask, y)

                pred = torch.argmax(pred_logits, dim=1)

                y_pred.append(pred.detach().cpu().numpy())
                y_true.append(y.cpu().numpy())

                total_loss += len(input_ids) * loss.detach().cpu().numpy()
                total_len += len(input_ids)

            avg_loss = total_loss / total_len
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)

            acc = np.sum(y_pred == y_true) / len(y_pred)

            precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes),
                                                                           zero_division=0)

        if previously_training:
            model.train()

        return avg_loss, precision, recall, fscore, acc


def log(val, file, print_output=False):
    with open(file, 'a') as f:
        f.write(val + '\n')
    if print_output:
        print(val)


def print_results(dataset_name, labels, log_file, print_output, loss, precision, recall, fscore, accuracy):
    fmt_header = "{:<20.20}"
    for _ in labels:
        fmt_header += "{:<13.10}"

    fmt_row = "{:<20.20}"
    for _ in labels:
        fmt_row += "{:<13.3f}"

    log(fmt_header.format(dataset_name.title() + " Scores:", *labels), log_file, print_output)
    log(fmt_row.format("Precision", *precision), log_file, print_output)
    log(fmt_row.format("Recall", *recall), log_file, print_output)
    log(fmt_row.format("F1", *fscore), log_file, print_output)
    log(dataset_name.title() + " Accuracy is {}".format(accuracy), log_file, print_output)
    log(dataset_name.title() + " Loss is {}".format(loss), log_file, print_output)
    log("", log_file, print_output)
