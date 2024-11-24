import os
import glob
import torch
from .Utils import *
import torch.nn as nn
import logging
import torch.nn.functional as F
from . import infer
import torcheval.metrics as ms
from .metrics import *

if is_notebook():
    from tqdm.notebook import tqdm as tqbar
else:
    from tqdm import tqdm as tqbar


class Trainer(Base):
    def __init__(
        self,
        model,
        data,
        max_epochs=200,
        gpus=None,
        gradient_clip_val=0,
        lr=0.1,
        weight_decay=0.01,
        save_model=False,
        load_previous=False,
        save_loss_threshhold=1,
        logger=None,
        verbosity=0,
        mfs: List[MetricsFrame] = []
    ):
        self.save_attr()
        if not gpus:
            self.gpus = get_gpus(-1)  # get all gpus by default
        self.tunable = ["lr", "weight_decay"]  # affects saved model name
        self.board = ProgressBoard(xlabel="epoch")
        self._best_loss = save_loss_threshhold  # save model loss threshold

    def prepare_optimizers(self, **kwargs):
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **kwargs,
        )

    def prepare_batch(self, batch):
        return batch  # Handled by DataParallel

        if self.gpus:
            return [a.to(self.device) for a in batch]  # Move batch to the first device

    @property
    def num_train_batches(self):
        return len(self.train_dataloader) if self.train_dataloader is not None else 0

    @property
    def num_val_batches(self):
        return len(self.val_dataloader) if self.val_dataloader is not None else 0

    def prepare_model(self, model):
        self._loaded = False
        self._model = model  # store a reference
        # Easy way to run on gpu, better to use the following
        # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        self.model = nn.DataParallel(model, self.gpus)

        if self.load_previous:
            params = self.__load_previous()
            if params is not None:
                self._loaded = True
                # if not self.gpus and params["gpus"]:
                #     self.model = self.model.module.to(
                #         "cpu"
                #     )  # unwrap from DataParallel if wrapped

        if not self._loaded:
            self.prepare_optimizers()
            self.first_epoch = 0

        # Inherit some overrideable functions/attributes
        if getattr(self, "loss", None) is None:
            self.loss = model.loss

        logging.debug(self.model.named_parameters())



    def prepare_metrics(self, mf=[]):
        # instantiate some default metric frames
        for mf in self.mfs:
            if mf.compute_every == -1:
                mf.compute_every = self.num_train_batches if mf.tra
        loss_metric_column = 
        if features in

    def init(self, model, data):
        self.train_dataloader, self.val_dataloader = data.get_dataloaders()
        self.prepare_model(model)
        self.prepare_metrics()

        # Configure graphical parameters
        self.board.xlim = self.max_epochs

    def fit(
        self,
        model,
        data,
    ):
        """Initialize model and data, and begins training

        Args:
            model (Module)
            data (DataModule)
            torchevals,batch_fun,pred_funs

        Returns:
            float: best loss
        """

        self.init(model, data)

        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        epoch_bar = tqbar(range(self.max_epochs), desc="Epochs progress", unit="Epoch")
        batch_fun = self._make_batch_fun

        for self.epoch in epoch_bar:
            epoch_loss = self._fit_epoch(batch_fun=batch_fun)
            epoch_bar.set_description(
                "Epochs progress [Loss: {:.3e}]".format(epoch_loss)
            )

        self.board.flush()

        l = self._best_loss
        self._best_loss = self.save_loss_threshhold
        return l

    def _fit_epoch(self, train_dataloader=None, val_dataloader=None, y_len=None):
        train_dataloader = train_dataloader or self.train_dataloader
        val_dataloader = val_dataloader or self.val_dataloader

        self.model.train()
        losses = 0
        vals = []
        for batch in train_dataloader:
            outputs = self.model(*batch[:-y_len])
            Y = batch[-y_len:]
            loss = self.loss(outputs, Y.to(self.device))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                losses += loss
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()

                for m in self.train_mfs:
                    m.update(outputs, Y, True)
                self.loss_mf.update(loss)
            self.train_batch_idx += 1

            self._debug()


        if val_dataloader is not None:
            self.model.eval()  # Set the model to evaluation mode, this disables training specific operations such as dropout and batch normalization
            for batch in val_dataloader:
                with torch.no_grad():
                    outputs = self.model(*batch[:-y_len])
                    Y = batch[-y_len:]
                    loss = self.loss(outputs, Y.to(self.device))

                    for m in self.val_mfs:
                        m.update(outputs, Y, False)
                self.val_batch_idx += 1


        epoch_loss = losses / self.num_train_batches

        if self.logger:
            self.logger(
                {"epoch": self.epoch, "loss": epoch_loss, **mean_of_dicts(vals)}
            )

        if epoch_loss <= self._best_loss:
            self._best_loss = epoch_loss
            if self.save_model:
                self.__save_model()
        return epoch_loss

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def plot(self, label, y, train):
        """Plot a point wrt epoch"""
        if train:
            if self.train_points_per_epoch == 0:  # use to disable plotting/storage
                return
            x = self.train_batch_idx / self.num_train_batches
            n = self.num_train_batches // self.train_points_per_epoch
        else:
            x = self.epoch + 1
            if self.num_val_batches == 0:
                return
            n = self.valid_points_per_epoch // self.num_val_batches

        # move to cpu
        if getattr(y, "device") not in ["cpu", None]:
            y = y.detach().cpu()
        else:
            y = y.detach()

        label = f"{'train_' if train else ''}{label}"
        self.board.draw_points(x, y, label, every_n=n)

    def training_step(self, batch):
        """Compute (and plot loss of a batch) during training step"""
        Y_hat = self.model(*batch[:-1])
        l = self.loss(Y_hat, batch[-1].to(self.device))
        self.plot("loss", l, train=True)
        return l

    # returns a dict
    def validation_step(self, batch):
        """Compute (and plot loss of a batch) during validation step"""
        Y_hat = self.model(*batch[:-1])
        l = self.loss(Y_hat, batch[-1].to(self.device))
        self.plot("loss", l, train=False)
        return {"val_loss", l}

    def eval(
        self,
        torchevals=[],
        batch_fun=None,
        pred_funs=None,
        dataloader=None,
        loss=False,
    ):
        """
        Evaluates the model on a given dataloader and computes the metrics and/or loss.

        Args:
            torchevals (list, optional): List of evaluation metrics or metric groups to be updated during evaluation.
                                        Example: [ms.torcheval.metrics.Cat()] or [[torcheval.metrics.BinaryAccuracy()], [torcheval.metrics.Cat()]]
                                        For a metric in group i, it is updated with m.update(pred_funs[i](outputs),  *Y)
            pred_funs (list, optional): List of prediction functions to be applied to the model outputs.
                                        By default, will apply model.pred to group 1 if defined. If torch_evals is longer, the groups use output directly: i.e. m.update(outputs, *Y)
            batch_fun (function, optional): Custom definition of function that is applied with batch_fun(outputs, *Y) to each batch. If not supplied, will update supplied torchevals using pred_funs, then output [predictions] or [predictions, loss].
            dataloader (DataLoader, optional): The DataLoader to iterate through during evaluation. If None,
                                                defaults to `self.val_dataloader`.
            loss (bool, optional): Whether to output batch_loss in batch_fun. Defaults to False. A custom loss function can also be supplied.

        Returns:
            tuple: List of metrics, as many as are output by batch_fun. Tensor type metrics are concatenated, while others are arrays of len(dataloader).
                updated metrics, and the second element is the computed loss if requested.
        """
        if getattr(self, "_model", None) is None:
            self.init()

            if loss is True:
                loss = self.loss

            if pred_funs is None:
                pred_funs = [getattr(self._model, "_pred", lambda x: x)]

            if batch_fun is None:
                batch_fun = infer.make_batch_fun(torchevals, pred_funs, loss)

        return infer.infer(
            self.model,
            dataloader or self.val_dataloader,
            batch_fun,
            device=self.device,
        )

    def _debuf(self):
        if self.verbosity > 5:
            for param in self.model.named_parameters():
                if param[1].grad is None:
                    print("No gradient for parameter:", param)
                elif torch.all(param[1].grad == 0):
                    print("Zero gradient for parameter:", param)

    @property
    def filename(self):
        # Filter and create the string for the tunable parameters that exist in self.p
        param_str = "__".join(
            [f"{k}_{v}" for k, v in self.p.items() if k in self.tunable]
        )
        return f"{self._model.filename}__{param_str}"

    def __save_model(self, params={}):
        with change_dir(self.save_path):
            filename = (
                self.filename
                + f"__epoch_{self.first_epoch}-{self.first_epoch+self.epoch}"
                + ".pth"
            )
            torch.save(
                {
                    "params": params,
                    "epoch": self.first_epoch
                    + self.epoch,  # save the epoch of the model
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                filename,
            )

    # load a previous model to train further
    def __load_previous(self):
        with change_dir(self.save_path):
            files = glob.glob(self._model.filename + "_epoch*.pth")
            # look for the most recent file
            files.sort(key=os.path.getmtime)
            if len(files) > 0:
                print("Found older file:", files[-1])
                print("Loading.....")
                checkpoint = torch.load(
                    files[-1], map_location=self.gpus[0] if self.gpus else "cpu"
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # continue on next epoch
                self.first_epoch = checkpoint["epoch"] + 1
                return checkpoint["params"]
            return None


sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "max_epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}


def tune(model, data, sweep_configuration=sweep_configuration):
    import wandb

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=model.filename)

    def objective():
        run = wandb.init()  # noqa: F841

        trainer = Trainer(logger=wandb.log, **wandb.config)
        return trainer.fit(model, data)

    # Start sweep job.
    wandb.agent(sweep_id, function=objective, count=4)
