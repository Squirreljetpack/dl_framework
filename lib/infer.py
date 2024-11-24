from .Utils import *
import torcheval.metrics as ms


if is_notebook():
    from tqdm.notebook import tqdm as tqbar
else:
    from tqdm import tqdm as tqbar


@torch.inference_mode()
def infer(model, dataloader, batch_fun=lambda x, y: x, y_len=1, device="cpu"):
    """
    Performs inference on a given model using a dataloader, updating batch-level metrics using the provided function.

    Args:
        model (torch.nn.Module): The model to run inference on.
        dataloader (DataLoader): The DataLoader to provide batches of data for inference.
        batch_fun (function, optional): A function that processes the model's predictions and the true values
                                        for each batch. Defaults to just returns the model output.
        y_len (int, optional): The number of elements in the target batch. Defaults to 1.
        device (str, optional): The device to compute metrics on, e.g., "cpu" or "cuda".

    Returns:
        tuple: List of metrics, as many as are output by batch_fun.
    """

    model_training = model.training

    model.eval()
    batch_metrics = []

    for _, batch in tqbar(enumerate(dataloader)):
        outputs = model(*batch[:-y_len]).to(device)
        Y = batch[-y_len:].to(device)
        computed = batch_fun(outputs, *Y)
        if len(batch_metrics) == 0:  # instantiate
            batch_metrics = [[m] for m in computed]
        else:
            for i, m in enumerate(computed):
                batch_metrics[i].append(m)

    model.train(model_training)

    return [
        torch.cat(m, dim=0) if isinstance(m, torch.Tensor) else m for m in batch_metrics
    ]


def make_batch_fun(torchevals, pred_funs, loss):
    pred_funs = k_level_list(pred_funs, k=1)

    torchevals = k_level_list(torchevals, k=2)
    pred_fun += [lambda x: x] * len(torchevals) - len(pred_fun)
    # preds = ms.Cat()

    def batch_fun(Y, Y_hat):
        for group, fun in zip(torchevals, pred_funs):
            ps = k_level_list(fun(Y))
            for m in group:
                m.update(*ps, Y_hat)
        if loss:
            return [ps, loss(Y, Y_hat)]
        else:
            return [ps]

    return batch_fun


def eval_classifier(trainer):
    metrics = [
        ms.BinaryConfusionMatrix(),
        ms.BinaryAccuracy(),
        ms.BinaryPrecision(),
        ms.BinaryRecall(),
        ms.BinaryF1Score(),
    ]
    trainer.eval(torchevals=[])

    return [m.compute() for m in metrics]


def loader_columns(dataloader, columns=slice(-1, None), cpu=True):
    outs = [batch[columns] for batch in dataloader]
    out = torch.cat(outs, dim=0)
    if cpu:
        return out.cpu().numpy()
    return out
