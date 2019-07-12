from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cv2
from utils import CFEVideoConf, image_resize
import torchreid
import sys
import os
import os.path as osp
import glob
import re
import warnings
from torchreid.data.datasets import ImageDataset

class NewDataset(ImageDataset):
    dataset_dir = 'new_dataset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data


torchreid.data.register_image_dataset('new_dataset', NewDataset)

# use your own dataset only
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='new_dataset'
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.cuda()
torchreid.utils.load_pretrained_weights(model, 'log/resnet50/model.pth.tar-120')

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

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/resnet50',
    max_epoch=60,
    eval_freq=10,
    visrank=True,
    visrank_topk=150,
    print_freq=10,
    test_only=True
)

query_result_dir = os.listdir("./log/resnet50/visrank-1/new_dataset")[0]
cams = os.listdir("./log/resnet50/visrank-1/new_dataset/" + query_result_dir)

for camid in cams:
    # Build video with overlapping images from model result
    cap = cv2.VideoCapture("videos/" + camid + ".mp4")
    save_path = 'output/' + camid + '.avi'
    frames_per_seconds = 24
    config = CFEVideoConf(cap, filepath=save_path, res='720p')
    out = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)

    img_path = 'cfe-coffee.jpg'
    images = glob.glob(osp.join("./log/resnet50/visrank-1/new_dataset/" + query_result_dir + "/" + camid, "*.jpg"))
    logo = cv2.imread(img_path, -1)
    watermark = image_resize(logo, height=50)
    #watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    #watermark = cv2.cvtColor(watermark, cv2.COLOR_GRAY2BGR)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)

    count = 0
    img_idx = 0
    while(True):
        count += 1
        pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_([-\d]+)')

        if img_idx >= len(images):
            _, _, _, fr = map(int, pattern.search(images[len(images) - 1]).groups())
        else:
            _, _, _, fr = map(int, pattern.search(images[img_idx]).groups())
            if fr == count:
                img_path = images[img_idx]
                img_idx += 1
                logo = cv2.imread(img_path, -1)
                watermark = image_resize(logo, height=50)
                watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        frame_h, frame_w, frame_c = frame.shape

        # overlay with 4 channels BGR and Alpha
        overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')

        watermark_h, watermark_w, watermark_c = watermark.shape
        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                if watermark[i,j][3] != 0:
                    offset = 10
                    h_offset = frame_h - watermark_h - offset
                    w_offset = frame_w - watermark_w - offset
                    # print(len(overlay))
                    # print(len(overlay[0]))
                    # print(h_offset + i)
                    # print(w_offset + j)
                    # print(len(watermark))
                    # print(len(watermark[0]))
                    # print(i)
                    # print(j)
                    overlay[h_offset + i, w_offset+ j] = watermark[i,j]

        cv2.addWeighted(overlay, 1, frame, 1, 0, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
