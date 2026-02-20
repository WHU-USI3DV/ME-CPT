_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8 # bs: total bs in all gpus
num_worker = 0
mix_prob = 0.8
empty_cache = False
enable_amp = True
weight =  "" # path to model weight
save_path="/home/luqizhang/work/data/PTV3TEST"
test_files="/home/luqizhang/work/data/MONI01/test.txt"
test_root="/home/luqizhang/work/data/MONI01/Test"

#resume = False  # whether to resume training process
# model settings
model = dict(
    type="DefaultSegmentorV3",
    num_classes=7,
    num_classes_seg=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=False,#true
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,  #true
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0),
        #dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    criteria_cd=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        #dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 100
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "CDDataset"
data_root = "/home/luqizhang/work/data/MONI01"

data = dict(
    num_classes=7,
    num_classes_seg=4,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split="Train",
        data_root=data_root,
        pre_split="Pre",
        radius=50,
        voxel_size=0.2,
        #grid=0.2,
        nameInPly="params",
        sample_num=6000,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
        ]
    ),
    val=dict(
        type=dataset_type,
        split="Val",
        data_root=data_root,
        pre_split="Pre",
        radius=50,
        voxel_size=0.2,
        #grid=0.2,
        nameInPly="params",
        sample_num=1000,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
        ]
    ),
    test=dict(
        type=dataset_type,
        split="Val",
        data_root=data_root,
        pre_split="Pre",
        radius=50,
        voxel_size=0.2,
        #grid=0.2,
        nameInPly="params",
        sample_num=1000,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
        ]
    ),
)
