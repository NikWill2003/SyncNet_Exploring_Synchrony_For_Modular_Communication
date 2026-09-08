import hydra
from hydra.core.hydra_config import HydraConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from src.core import (
    Config,
    build_callbacks,
    build_dataloaders,
    build_loss_fn,
    build_lr_scheduler,
    build_model,
    build_optim,
    register_configs,
)
from src.training import Trainer, accelerate_init_wandb
from src.utils import set_seed, set_torch_config

register_configs()

logger = get_logger(__name__)


def build_components(cfg: Config) -> dict:

    out_dir = HydraConfig.get().runtime.output_dir

    set_seed(cfg.train.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.grad_accum,
        log_with='wandb' if cfg.wandb.enabled else None,
        project_config=ProjectConfiguration(project_dir=out_dir),
    )

    device = str(accelerator.device)
    set_torch_config(device)

    dataloaders = build_dataloaders(cfg, device)
    model = build_model(cfg)

    if cfg.train.compile_model:
        model.compile(mode=cfg.train.compile_mode)

    optimiser = build_optim(model, cfg.optim)
    scheduler = build_lr_scheduler(optimiser, cfg.train.n_steps, cfg.optim)
    loss_fn = build_loss_fn(cfg)
    callbacks = build_callbacks(cfg, model)

    if cfg.wandb.enabled:
        if cfg.wandb.project_name is None:
            raise ValueError('project name must be specified when wandb is enabled')

        accelerate_init_wandb(cfg, accelerator, out_dir, model)

    return dict(
        out_dir=out_dir,
        model=model,
        dataloaders=dataloaders,
        optimiser=optimiser,
        scheduler=scheduler,
        accelerator=accelerator,
        callbacks=callbacks,
        loss_fn=loss_fn,
    )


@hydra.main(config_path='../conf', config_name='config', version_base='1.3')
def main(cfg: Config) -> None:

    components = build_components(cfg)

    try:
        trainer = Trainer(cfg=cfg, logger=logger, **components)
        trainer.train()
    finally:
        components['accelerator'].end_training()


if __name__ == '__main__':
    main()
