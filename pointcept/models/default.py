import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
import torch

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultSegmentorV3(nn.Module):
    def __init__(
        self,
        num_classes,
        num_classes_seg,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        criteria_cd=None,
    ):
        super().__init__()
        
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        #   dualtask
        self.seg_head_seg = (
            nn.Linear(backbone_out_channels, num_classes_seg)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.criteria_cd = build_criteria(criteria_cd)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        

        seg_logits = self.seg_head(point.feat)
        #dualtask
        seg_logits_seg = self.seg_head_seg(point.feat)
     
        seg=input_dict["segment_cd"]
        
        
        if self.training:
            #dualtask
            loss = self.criteria(seg_logits_seg, input_dict["segment"])+ self.criteria_cd(seg_logits, input_dict["segment_cd"])
           
            return dict(loss=loss)
        # eval
        elif "segment_cd" in input_dict.keys():
            #dualtask
            loss = self.criteria(seg_logits_seg, input_dict["segment"])+ self.criteria_cd(seg_logits, input_dict["segment_cd"])
            return dict(loss=loss, seg_logits=seg_logits,feat=point.feat)
        # # test
        else:
            return dict(seg_logits=seg_logit,feat=point.feat)

@MODELS.register_module()
#single task
class DefaultSegmentorV4(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria_cd=None,
    ):
        super().__init__()
        
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        
        self.backbone = build_model(backbone)
        self.criteria_cd = build_criteria(criteria_cd)
        

    def forward(self, input_dict):



        point = Point(input_dict)
        point = self.backbone(point)

        seg_logits = self.seg_head(point.feat)
       
        # train
        if self.training:
            loss = self.criteria_cd(seg_logits, input_dict["segment_cd"])
            return dict(loss=loss)
        # eval
        elif "segment_cd" in input_dict.keys():
            loss = self.criteria_cd(seg_logits, input_dict["segment_cd"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = self.backbone(input_dict)
  
        cls_logits = self.cls_head(point["feat"])

        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"].long())
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"].long())
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
