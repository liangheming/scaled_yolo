import math
import torch
from torch import nn
from nets.activations import MemoryEfficientMish
from nets.commons import CBR, SPPCSP, BottleNeckCSP, BottleNeckCSP2, width_grow, depth_grow, model_scale
from losses.yolo_loss import YOLOv5Loss
from utils.boxs_utils import xyxy2xywh, box_iou, xywh2xyxy, clip_coords
from torchvision.ops import nms


def non_max_suppression(prediction,
                        conf_thresh=0.1,
                        iou_thresh=0.6,
                        merge=False,
                        agnostic=False,
                        multi_label=True,
                        max_det=300):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    xc = prediction[..., 4] > conf_thresh  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    redundant = True  # require redundant detections
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thresh).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thresh]

        # Filter by class

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thresh  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]

    return output


class YOLOV4BackBone(nn.Module):
    def __init__(self,
                 in_channel=3,
                 depth_multiples=0.33,
                 width_multiples=0.50,
                 act=MemoryEfficientMish(),
                 extra_len=3):
        super(YOLOV4BackBone, self).__init__()
        channel_32 = width_grow(32, width_multiples)
        channel_64 = width_grow(64, width_multiples)
        channel_128 = width_grow(128, width_multiples)
        channel_256 = width_grow(256, width_multiples)
        channel_512 = width_grow(512, width_multiples)
        channel_1024 = width_grow(1024, width_multiples)

        self.stem = nn.Sequential(
            CBR(in_channel, channel_32, 3, 1, act=act),
            CBR(channel_32, channel_64, 3, 2, act=act),
            BottleNeckCSP(channel_64, channel_64, depth_grow(1, depth_multiples), act=act),
            CBR(channel_64, channel_128, 3, 2, act=act),
            BottleNeckCSP(channel_128, channel_128, depth_grow(3, depth_multiples), act=act)
        )

        self.layer3 = nn.Sequential(
            CBR(channel_128, channel_256, 3, 2, act=act),
            BottleNeckCSP(channel_256, channel_256, depth_grow(15, depth_multiples), act=act)
        )

        self.layer4 = nn.Sequential(
            CBR(channel_256, channel_512, 3, 2, act=act),
            BottleNeckCSP(channel_512, channel_512, depth_grow(15, depth_multiples), act=act)
        )

        self.layer5 = nn.Sequential(
            CBR(channel_512, channel_1024, 3, 2, act=act),
            BottleNeckCSP(channel_1024, channel_1024, depth_grow(7, depth_multiples), act=act)
        )
        self.extra = list()
        for _ in range(extra_len - 3):
            self.extra.append(
                nn.Sequential(
                    CBR(channel_1024, channel_1024, 3, 2, act=act),
                    BottleNeckCSP(channel_1024, channel_1024, depth_grow(7, depth_multiples), act=act)
                )
            )
        self.extra = None if len(self.extra) == 0 else nn.ModuleList(self.extra)
        self.output_channels = [channel_256,
                                channel_512,
                                channel_1024] + [channel_1024] * (extra_len - 3)

    def forward(self, x):
        ret_list = list()
        x = self.stem(x)
        x = self.layer3(x)
        ret_list.append(x)
        x = self.layer4(x)
        ret_list.append(x)
        x = self.layer5(x)
        ret_list.append(x)
        if self.extra is not None:
            for layer in self.extra:
                x = layer(x)
                ret_list.append(x)
        return ret_list


class YOLOV4Neck(nn.Module):
    def __init__(self, in_channels, blocks=1, k=(5, 9, 13), act=MemoryEfficientMish()):
        super(YOLOV4Neck, self).__init__()
        inner_channels = [i // 2 for i in in_channels]
        self.d_layer = list()
        self.l_layer = list()
        self.b_layer = list()
        self.spp = SPPCSP(in_channels[-1], inner_channels[-1], act=act, k=k)
        for i in range(len(inner_channels) - 1, 0, -1):
            self.d_layer.append(
                CBR(inner_channels[i], inner_channels[i - 1], 1, 1, act=act)
            )
            self.l_layer.append(
                CBR(in_channels[i - 1], inner_channels[i - 1], 1, 1, act=act)
            )
            self.b_layer.append(
                BottleNeckCSP2(in_channels[i - 1], inner_channels[i - 1], blocks, act=act)
            )
        self.o_layer = list()
        self.u_layer = list()
        self.r_layer = list()
        for j in range(0, len(in_channels) - 1):
            self.u_layer.append(CBR(inner_channels[j], inner_channels[j + 1], 3, 2, act=act))
            self.o_layer.append(CBR(inner_channels[j], in_channels[j], 3, 1, act=act))
            self.r_layer.append(BottleNeckCSP2(in_channels[j + 1], inner_channels[j + 1], blocks, act=act))
        self.o_layer.append(CBR(inner_channels[-1], in_channels[-1], 3, 1, act=act))

        self.d_layer = nn.ModuleList(self.d_layer)
        self.l_layer = nn.ModuleList(self.l_layer)
        self.b_layer = nn.ModuleList(self.b_layer)

        self.o_layer = nn.ModuleList(self.o_layer)
        self.u_layer = nn.ModuleList(self.u_layer)
        self.r_layer = nn.ModuleList(self.r_layer)

    def forward(self, x):
        x[-1] = self.spp(x[-1])
        for i in range(1, len(x)):
            l = self.l_layer[i - 1](x[-(i + 1)])
            d = nn.UpsamplingNearest2d(size=l.shape[-2:])(self.d_layer[i - 1](x[-i]))
            x[-(i + 1)] = self.b_layer[i - 1](torch.cat([l, d], dim=1))
        for j in range(0, len(x) - 1):
            o = self.o_layer[j](x[j])
            u = self.u_layer[j](x[j])
            x[j + 1] = self.r_layer[j](torch.cat([u, x[j + 1]], dim=1))
            x[j] = o
        x[-1] = self.o_layer[-1](x[-1])
        return x


class YOLOV4Head(nn.Module):
    def __init__(self, in_channels, anchors, strides, num_cls=80):
        super(YOLOV4Head, self).__init__()
        assert len(in_channels) == len(anchors) == len(strides)
        self.num_cls = num_cls
        self.output_num = num_cls + 5
        self.strides = strides
        self.anchors = anchors
        self.in_channels = in_channels
        self.layer_num = len(self.anchors)
        self.anchor_per_grid = len(self.anchors[0]) // 2
        self.grids = [torch.zeros(1)] * self.layer_num
        # [layer_num,3,2]
        a = torch.tensor(self.anchors, requires_grad=False).float().view(self.layer_num, -1, 2)
        #  [layer_num,3,2]
        normalize_anchors = a / torch.tensor(strides, requires_grad=False).float().view(self.layer_num, 1, 1)
        self.register_buffer("normalize_anchors", normalize_anchors.clone())
        # [layer_num,1,3,1,1,2]
        self.register_buffer("anchor_grid", a.clone().view(self.layer_num, 1, -1, 1, 1, 2))
        self.heads = nn.ModuleList(
            nn.Conv2d(x, self.output_num * self.anchor_per_grid, 1) for x in in_channels
        )
        for mi, s in zip(self.heads, strides):  # from
            b = mi.bias.view(self.anchor_per_grid, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8. / (640. / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (self.num_cls - 0.99))  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xs):
        z = list()
        assert len(xs) == self.layer_num
        for i in range(self.layer_num):
            xs[i] = self.heads[i](xs[i])
            bs, _, ny, nx = xs[i].shape
            # [bs,anchor_num,h,w,output_num]
            xs[i] = xs[i].view(bs, self.anchor_per_grid, self.output_num, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference
                if self.grids[i].shape[2:4] != xs[i].shape[2:4]:
                    self.grids[i] = self._make_grid(nx, ny).to(xs[i].device)
                # grid: bs,anchor_per_grid,ny,nx,2
                # xs[i]:bs,anchor_per_grid,ny,nx,output
                y = xs[i].sigmoid()

                # ys[i] [bs, anchor_num, h, w, output_num]
                # anchor_grid[i] = [1,3,1,1,2]
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grids[i]) * self.strides[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.output_num))
        return (xs, self.normalize_anchors) if self.training else torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


default_cfg = {
    "in_channel": 3,
    "num_cls": 80,
    "scale_name": "s",
    "strides": [8., 16., 32.],
    "anchors": [
        [12, 16, 19, 36, 40, 28],
        [36, 75, 76, 55, 72, 146],
        [142, 110, 192, 243, 459, 401]
    ],
    "k": (5, 9, 13),
    'act': MemoryEfficientMish(),
    "balance": {
        3: [4.0, 1.0, 0.4],
        4: [4.0, 1.0, 0.4, 0.1],
        5: [4.0, 1.0, 0.5, 0.4, 0.1]
    },
    "ratio_thresh": 4.,
    "expansion_bias": 0.5,
    "cls_pw": 1.0,
    "obj_pw": 1.0,
    "iou_type": "ciou",
    "coord_type": "xywh",
    "iou_ratio": 1.0,
    "iou_weights": 0.05,
    "cls_weights": 0.5,
    "obj_weights": 1.0,

    "conf_thresh": 0.001,
    "iou_thresh": 0.6,
    "merge": True,
    "max_det": 300
}


class YOLOV4(nn.Module):
    def __init__(self, **kwargs):
        super(YOLOV4, self).__init__()
        self.cfg = {**default_cfg, **kwargs}
        extra_len = len(self.cfg['strides'])
        assert extra_len >= 3
        assert len(self.cfg['strides']) == len(self.cfg['anchors'])
        depth_multiples, width_multiples = model_scale(self.cfg['scale_name'])
        act = self.cfg['act']
        self.backbones = YOLOV4BackBone(self.cfg['in_channel'],
                                        depth_multiples=depth_multiples,
                                        width_multiples=width_multiples,
                                        act=act,
                                        extra_len=extra_len)
        out_channels = self.backbones.output_channels
        self.neck = YOLOV4Neck(in_channels=out_channels,
                               blocks=depth_grow(3, depth_multiples),
                               k=self.cfg['k'],
                               act=act)
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True
        self.head = YOLOV4Head(in_channels=out_channels,
                               num_cls=self.cfg['num_cls'],
                               strides=self.cfg['strides'],
                               anchors=self.cfg['anchors'])

        self.loss = YOLOv5Loss(
            ratio_thresh=self.cfg['ratio_thresh'],
            expansion_bias=self.cfg['expansion_bias'],
            layer_balance=self.cfg['balance'][extra_len],
            cls_pw=self.cfg['cls_pw'],
            obj_pw=self.cfg['obj_pw'],
            iou_type=self.cfg['iou_type'],
            coord_type=self.cfg['coord_type'],
            iou_ratio=self.cfg['iou_ratio'],
            iou_weights=self.cfg['iou_weights'],
            cls_weights=self.cfg['cls_weights'],
            obj_weights=self.cfg['obj_weights']
        )

    def forward(self, x, targets=None):
        """
        :param x:
        :param targets:
        :return:
        """
        _, _, h, w = x.shape
        x = self.head(self.neck(self.backbones(x)))
        ret = dict()
        if self.training:
            assert targets is not None
            targets[:, 2:] = xyxy2xywh(targets[:, 2:])
            targets[:, [2, 4]] = targets[:, [2, 4]] / w
            targets[:, [3, 5]] = targets[:, [3, 5]] / h
            predicts, norm_anchors = x
            loss_cls, loss_box, loss_obj, match_num = self.loss(predicts, targets, norm_anchors)
            ret['cls_loss'] = loss_cls
            ret['box_loss'] = loss_box
            ret['obj_loss'] = loss_obj
            ret['match_num'] = match_num
        else:
            predicts = non_max_suppression(x,
                                           conf_thresh=self.cfg['conf_thresh'],
                                           iou_thresh=self.cfg['iou_thresh'],
                                           merge=self.cfg['merge'],
                                           max_det=self.cfg['max_det'])
            for predict in predicts:
                if predict is not None:
                    clip_coords(predict, (h, w))
            ret['predicts'] = predicts
        return ret


if __name__ == '__main__':
    input_tensor = torch.rand(size=(1, 3, 640, 640))
    net = YOLOV4()
    out = net(input_tensor)
    out, norm = out
    for i in out:
        print(i.shape)
    print(norm.shape)
