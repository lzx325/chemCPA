from pathlib import Path
from pprint import pprint

from sacred.observers import FileStorageObserver
from seml.config import generate_configs, read_config
from seml.observers import add_to_file_storage_observer

from chemCPA.experiments_run import ExperimentWrapper
import chemCPA.experiments_run as er
from tensorboardX import SummaryWriter
if __name__ == "__main__":
    er.ex.observers.append(FileStorageObserver.create('my_runs'))
    exp = ExperimentWrapper(init_all=False)

    # this is how seml loads the config file internally
    config_fp = "manual_run.yaml"
    assert Path(
        config_fp
    ).exists(), "config file not found"
    seml_config, slurm_config, experiment_config = read_config(
        config_fp
    )
    # we take the first config generated
    configs = generate_configs(experiment_config)
    if len(configs) > 1:
        print("Careful, more than one config generated from the yaml file")
    args = configs[0]
    pprint(args)
    exp.seed = 1337
    # loads the dataset splits
    exp.init_dataset(**args["dataset"])

    exp.init_drug_embedding(embedding=args["model"]["embedding"])
    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        enable_cpa_mode=args["model"]["enable_cpa_mode"],
        pretrained_model_path=args["model"]["pretrained_model_path"],
        pretrained_model_hashes=args["model"]["pretrained_model_hashes"],
    )
    # setup the torch DataLoader
    exp.update_datasets()
    print("current run",er.ex.current_run)
    writer = SummaryWriter()
    try:
        exp.train(**args["training"], writer = writer)
    finally:
        writer.close()
