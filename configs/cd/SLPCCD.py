_base_ = ["../_base_/default_runtime.py"]
#gaoi gridsize

# misc custom setting
batch_size =4# bs: total bs in all gpus
num_worker = 0
mix_prob = 0.8
empty_cache = False
enable_amp = False
#weight =  "/home/luqizhang/work/data/checkpoints/SLPCCD/model/model_best.pth" # path to model weight
save_path="/home/luqizhang/work/data/SHREC2020-CD/vis-SHREC-ours"
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegBiEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
test=dict(type="SemSegBiTester", verbose=True)

#test_mode
save_file="/home/luqizhang/work/data/SHREC2020-CD/vis-SHREC-ours"
test_type="SemSegBiTester"
test_files="/home/luqizhang/work/data/SHREC2020-CD/Test.txt"
test_root="/home/luqizhang/work/data/SHREC2020-CD/Test"

resume = False  # whether to resume training processL
# model settings
model = dict(
    type="DefaultSegmentorV4",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1-b",
        in_channels=7,
        order=("z", "z-trans"), 
        grid_size=0.03,
        stride=(2, 2,2, 2),
        enc_depths=(2, 2, 2, 6, 2),#(2, 2, 2, 6, 2)
        enc_channels=(64, 64, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(512, 512, 512,512, 512),
        dec_depths=(2, 2,2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(512, 512,  512,512),
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
    criteria_cd=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),#,weight=[0.1,0.45,0.45]
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 200
eval_epoch=200

optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
# dataset settings
dataset_type = "CDDatasetStreet"
data_root = "/home/luqizhang/work/data/SHREC2020-CD"

data = dict(
    num_classes=4,
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
        pre_split="PRE",
        radius=25,
        voxel_size=0.03,
        #grid=0.2,
        nameInPly="vertex",
        sample_num=6000,
        # transform=[
        #     dict(
        #         type="GridSample",
        #         grid_size=0.05,
        #         hash_type="fnv",
        #         mode="train",
        #         return_grid_coord=True,
        #     ),
        # ]
    ),
    val=dict(
        type=dataset_type,
        split="Val",
        data_root=data_root,
        pre_split="PRE",
        radius=25,
        voxel_size=0.03,
        #grid=0.2,
        nameInPly="vertex",
        sample_num=-1,
        # transform=[
        #     dict(
        #         type="GridSample",
        #         grid_size=0.05,
        #         hash_type="fnv",
        #         mode="train",
        #         return_grid_coord=True,
        #     ),
        # ]
    ),
    test=dict(
        type=dataset_type,
        split="Test",
        data_root=data_root,
        pre_split="Pre",
        radius=50,
        voxel_size=0.2,
        #grid=0.2,
        nameInPly="vertex",
        sample_num=1000,
    ),
)
