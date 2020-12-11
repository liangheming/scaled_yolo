import torch
from nets.yolov4 import YOLOV4
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from utils.boxs_utils import kmean_anchors

if __name__ == '__main__':
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=False,
                           augments=True,
                           remove_blank=True,
                           max_thresh=640
                           )
    dataset.data_list = dataset.box_info_list
    kmean_anchors(dataset)
    # 9,14,  26,19,  20,44,  50,39,  45,90,  97,81,  96,195,  204,152,  325,344
    # 11,12,  15,34,  36,25,  35,61,  80,56,  66,136,  160,121,  159,269,  394,312
    # dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=dataset.collect_fn)
    # net = YOLOV4()
    # for img_input, targets, target_len in dataloader:
    #     out = net(img_input, targets)
    #     break
