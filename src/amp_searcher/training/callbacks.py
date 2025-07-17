from pytorch_lightning.callbacks import Callback


class GradientMonitor(Callback):
    """
    PyTorch Lightning Callback to log gradient norms to TensorBoard.
    """

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int, *args, **kwargs
    ):
        if trainer.logger:
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    trainer.logger.experiment.add_histogram(
                        f"gradients/{name}", param.grad, trainer.global_step
                    )
                    trainer.logger.experiment.add_scalar(
                        f"gradients_norm/{name}",
                        param.grad.norm(),
                        trainer.global_step,
                    )

    def on_before_optimizer_step(
        self,
        trainer,
        pl_module,
        optimizer,
    ):
        # Log total gradient norm before optimizer step
        if trainer.logger:
            total_norm = 0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            trainer.logger.experiment.add_scalar(
                "gradients_norm/total_norm", total_norm, trainer.global_step
            )
