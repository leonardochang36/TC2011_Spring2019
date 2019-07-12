import sys
import os
import os.path as osp
import warnings

import torch
import torch.nn as nn

from default_parser import (
    init_parser, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights
)


parser = init_parser()
args = parser.parse_args()


def build_datamanager(args):
    if args.app == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(args))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(args))


def build_engine(args, datamanager, model, optimizer, scheduler):
    if args.app == 'image':
        if args.loss == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth
            )
        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=args.margin,
                weight_t=args.weight_t,
                weight_x=args.weight_x,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth
            )

    else:
        if args.loss == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth,
                pooling_method=args.pooling_method
            )
        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=args.margin,
                weight_t=args.weight_t,
                weight_x=args.weight_x,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth
            )

    return engine


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = (torch.cuda.is_available() and not args.use_cpu)
    log_name = 'test.log' if args.evaluate else 'train.log'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))
    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        torch.backends.cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    datamanager = build_datamanager(args)
    model = torchreid.models.build_model(
        name=args.arch,
        num_classes=datamanager.num_train_pids,
        loss=args.loss.lower(),
        pretrained=(not args.no_pretrained),
        use_gpu=use_gpu
    )

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        **optimizer_kwargs(args)
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        **lr_scheduler_kwargs(args)
    )

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    print('Building {}-engine for {}-reid'.format(args.loss, args.app))
    engine = build_engine(args, datamanager, model, optimizer, scheduler)

    engine.run(**engine_run_kwargs(args))


if __name__ == '__main__':
    main()
