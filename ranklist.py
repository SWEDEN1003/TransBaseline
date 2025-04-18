import logging
import os
import torch
from torch.utils.data import DataLoader
import cv2
import torchvision.transforms as T
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
import random
import numpy as np
import argparse
import scipy.io
from config import cfg
from tqdm import tqdm
from datasets.bases import ImageDataset
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.vehicleid import VehicleID
from datasets.veri import VeRi

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def blend_cam(image, cam, img_name, save_path):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    a, b = cam.min(), cam.max()
    cam = (cam - a ) / (b - a)
    cam = (cam * 255.).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blend = image * 0.5 + heatmap * 0.5
    cv2.imwrite(f"{save_path}/{img_name}", np.uint8(blend))

def test_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, img_path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_path

def do_train(cfg,
            model,
            query_loader,
            galler_loader,
            num_query, local_rank):
    device = "cuda"
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model.eval()
    savepath = cfg.OUTPUT_DIR
    os.makedirs(savepath, exist_ok=True)


    query_featslist = []
    query_labellist = []
    query_camidlist = []
    query_img_pathlist = []

    gallery_featslist = []
    gallery_labellist = []
    gallery_camidlist = []
    gallery_img_pathlist = []


    for img, pid, camids, target_view, imgpath in tqdm(query_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            query_featslist.append(feat)
            query_labellist += list(pid.numpy())
            query_camidlist += list(camids.data.cpu().numpy())
            query_img_pathlist += list(imgpath)

    query_feats = torch.cat(query_featslist, dim=0)
    query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2)
    query_feats = query_feats.data.cpu()


    for img, pid, camids, target_view, imgpath in tqdm(galler_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            gallery_featslist.append(feat)
            gallery_labellist += list(pid.numpy())
            gallery_camidlist += list(camids.data.cpu().numpy())
            gallery_img_pathlist += list(imgpath)

    gallery_feats = torch.cat(gallery_featslist, dim=0)
    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)
    gallery_feats = gallery_feats.data.cpu()

    result = {
        "gallery_f": gallery_feats.numpy(),
        "gallery_label": gallery_labellist,
        "gallery_cam": gallery_camidlist,
        "gallery_img_paths": gallery_img_pathlist,
        "query_f": query_feats.numpy(),
        "query_label": query_labellist,
        "query_cam": query_camidlist,
        "query_img_paths": query_img_pathlist,
    }

    scipy.io.savemat(f"{savepath}/pytorch_result.mat", result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    test_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    query_set = ImageDataset(dataset.query, test_transforms, rank_list=True)
    galler_set = ImageDataset(dataset.gallery, test_transforms, rank_list=True)

    query_loader = DataLoader(
            query_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=test_collate_fn
        )
    galler_loader = DataLoader(
            galler_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=test_collate_fn
        )

    do_train(
        cfg,
        model,
        query_loader,
        galler_loader,
        num_query, args.local_rank
    )
