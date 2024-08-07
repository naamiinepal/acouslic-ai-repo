from typing import List, Tuple, Dict, Any
import hydra
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from src.utils.instantiators import instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.utils import extras, task_wrapper
from src.utils import instantiate_callbacks

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """evaluates given checkpoint on a datamodule testset.
    
    This method is wrapped in optional @task_wrapper decorator, that controls the behaviour during failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """

    # TODO: turn-of checkpoint checking for now (testing the pipeline) 
    # assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers..")
    logger: List = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)



    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "callbacks": callbacks,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # TODO: read the actual checkpoint once available
    trainer.predict(model=model,datamodule=datamodule,ckpt_path=None)


    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """ Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    
    evaluate(cfg)

if __name__ == "__main__":
    main()