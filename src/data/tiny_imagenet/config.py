from kaitorch.utils import Configer


def config_(cfg: Configer) -> Configer:
    if 'default' == cfg.data.num_category:
        cfg.data.num_category = 200

    return cfg
