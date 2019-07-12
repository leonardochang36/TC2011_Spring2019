import torchreid

datamanager = torchreid.data.VideoDataManager(
    root='reid-data',
    sources='mars',
    height=256,
    width=128,
    batch_size=1,
    seq_len=15,
    sample_method='evenly'
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.VideoSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    pooling_method=True
)

engine.run(
    save_dir='log/resnet50/video',
    max_epoch=30,
    eval_freq=1,
    print_freq=10,
    test_only=False
)
