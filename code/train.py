"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/simulation2surgery.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# Setup model and data loader
trainer = MUNIT_Trainer(config)
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
random.seed(1)
train_a = []
train_b = []
test_a = []
test_b = []
for i in range(display_size):
    train_a.append( train_loader_a.dataset[random.randrange(len(train_loader_a.dataset))] )
    train_b.append( train_loader_b.dataset[random.randrange(len(train_loader_b.dataset))] )
    test_a.append( test_loader_a.dataset[random.randrange(len(test_loader_a.dataset))] )
    test_b .append( test_loader_b.dataset[random.randrange(len(test_loader_b.dataset))] )
train_display_images_a = torch.stack(train_a).cuda()
train_display_images_b = torch.stack(train_b).cuda()
test_display_images_a = torch.stack(test_a).cuda()
test_display_images_b = torch.stack(test_b).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# Backup copy of current settings and scripts:
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
runPath = os.path.dirname(os.path.realpath(__file__))
for f in os.listdir(runPath):
    if f.endswith(".py"):
        shutil.copy( os.path.join(runPath, f), os.path.join(output_directory, f) ) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            #trainer.seg_update(images_a, labels_a, config)
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()
        trainer.update_learning_rate()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
            del test_image_outputs, train_image_outputs

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_current')
            write_2images(train_image_outputs, display_size, image_directory, 'train_current')
            del test_image_outputs, train_image_outputs

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
