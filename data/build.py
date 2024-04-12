import torch
from data.datasets import QBDataset, GF2Dataset, WV3Dataset, GF2LDataset, SPOTDataset, QBSXDataset, FRDataset, RRDataset, RRHPDataset
from data.concat_datasets import ConcatDataset


def build_concat_loader(config):
    if config.TRAIN.LWFS:
        syn_train_set = build_dataset(mode="train", type="RR", config=config)
        # syn_val_set = build_dataset(mode="val", type="RR", config=config)
        real_train_set = build_dataset(mode="train", type="FR", config=config)
        # real_val_set = build_dataset(mode="val", type="FR", config=config)
        train_set = ConcatDataset(syn_train_set, real_train_set)
        # val_set = ConcatDataset(syn_val_set, real_val_set)
        val_set = build_dataset(mode="val", type="FR", config=config)
    else:
        train_set = build_dataset(mode="train", type="FR", config=config)
        val_set = build_dataset(mode="val", type="FR", config=config)
    data_loader_train = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.DATA.BATCH_SIZE, shuffle=True,
        num_workers=config.DATA.NUM_WORKERS)

    data_loader_val = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=config.DATA.NUM_WORKERS)

    return data_loader_train, data_loader_val


def build_loader(config):
    train_set = build_dataset(mode="train", type=config.TRAIN.TYPE, config=config)
    val_set = build_dataset(mode="val", type=config.TRAIN.TYPE, config=config)

    data_loader_train = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.DATA.BATCH_SIZE, shuffle=True,
        num_workers=config.DATA.NUM_WORKERS)

    data_loader_val = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=config.DATA.NUM_WORKERS)

    return data_loader_train, data_loader_val


def build_dataset(mode, type, config):
    if config.DATA.DATASET == "qb":
        if type == "FR":
            dataset = FRDataset(config, mode=mode, data_dir=r"F:\ResearchData\dataset\QBFR128F/")
        elif type == "RRHP":
            dataset = RRHPDataset(config, mode=mode, data_dir=r"F:\ResearchData\dataset\QB128F/")
        else:
            dataset = RRDataset(config, mode=mode, data_dir=r"F:\ResearchData\dataset\QB128F/")

        config.defrost()
        config.MODEL.NUM_MS_BANDS = 4
        config.DATA.TEST_SET_PATH = r"F:\ResearchData\dataset\QB\forpresentation/"
        config.DATA.TESTSET_RR_PATH = r"F:\ResearchData\dataset\QB128F/"
        config.DATA.TESTSET_FR_PATH = r"F:\ResearchData\dataset\QBFR128F/"
        config.freeze()
    elif config.DATA.DATASET == "gf2":
        dataset = RRDataset(config, mode=mode, data_dir=r"F:\ResearchData\dataset\GF2128F/")

        config.defrost()
        config.MODEL.NUM_MS_BANDS = 4
        config.DATA.TEST_SET_PATH = r"F:\ResearchData\dataset\GF2\forpresentation/"
        config.DATA.TESTSET_RR_PATH = r"F:\ResearchData\dataset\GF2128F/"
        config.DATA.TESTSET_FR_PATH = r"F:\ResearchData\dataset\GF2FR128F/"
        config.freeze()
    elif config.DATA.DATASET == "wv3":
        if type == "FR":
            dataset = FRDataset(config, mode=mode, data_dir=r"F:\ResearchData\dataset\WV3\forpresentation/")
        elif type == "RRHP":
            dataset = RRHPDataset(config, mode=mode, data_dir=r"F:\ResearchData\dataset\WV3128F/")
        else:
            dataset = RRDataset(config, mode=mode, data_dir=r"F:\ResearchData\dataset\WV3FR128F/")

        config.defrost()
        config.MODEL.NUM_MS_BANDS = 8
        config.DATA.TEST_SET_PATH = r"F:\ResearchData\dataset\WV3\forpresentation//"
        config.DATA.TESTSET_RR_PATH = r"F:\ResearchData\dataset\WV3128F/"
        config.DATA.TESTSET_FR_PATH = r"F:\ResearchData\dataset\WV3FR128F/"
        config.freeze()
    elif config.DATA.DATASET == "gf2l":
        dataset = GF2LDataset(config, mode=mode)

        config.defrost()
        config.MODEL.NUM_MS_BANDS = 4
        config.DATA.TEST_SET_PATH = r"F:\ResearchData\dataset\GF2L\forpresentation/"
        config.freeze()
    elif config.DATA.DATASET == "spot":
        dataset = SPOTDataset(config, mode=mode)

        config.defrost()
        config.MODEL.NUM_MS_BANDS = 3
        config.DATA.TEST_SET_PATH = r"F:\ResearchData\dataset\SPOT2\forpresentation/"
        config.freeze()
    elif config.DATA.DATASET == "qbsx":
        dataset = QBSXDataset(config, mode=mode)

        config.defrost()
        config.MODEL.NUM_MS_BANDS = 3
        config.DATA.TEST_SET_PATH = r"F:\ResearchData\dataset\QBSX\forpresentation/"
        config.freeze()
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(config.DATA.DATASET))

    return dataset
