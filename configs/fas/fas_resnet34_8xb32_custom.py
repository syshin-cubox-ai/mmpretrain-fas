_base_ = [
    '../resnet/resnet34_8xb32_in1k.py'
]

randomness = dict(deterministic=True, seed=0)

# Dataset
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2,
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    std=[
        127.5,
        127.5,
        127.5,
    ],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    # batch_size=32,
    dataset=dict(
        type=dataset_type,
        data_root='../../data/mmpretrain_custom',
        data_prefix='train',
        with_label=True,
        pipeline=train_pipeline,
        _delete_=True
    )
)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs'),
]
val_dataloader = dict(
    # batch_size=32,
    dataset=dict(
        type=dataset_type,
        data_root='../../data/mmpretrain_custom',
        data_prefix='val',
        with_label=True,
        pipeline=val_pipeline,
        _delete_=True
    ),
)
val_evaluator = [
    dict(type='ACER'),
]
test_pipeline = val_pipeline
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Model
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth',
            prefix='backbone',
        )
    ),
    head=dict(num_classes=2),
)
