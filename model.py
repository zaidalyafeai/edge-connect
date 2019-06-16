# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect

class EdgeConnect_MODEL():
    def __init__(self, opts):
        print('loading configuration...')
        config = self.load_config()
        print('done.')
        # cuda visble devices
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

        # init device
        if torch.cuda.is_available():
            config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            config.DEVICE = torch.device("cpu")

        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)
        # initialize random seed
        torch.manual_seed(10)
        torch.cuda.manual_seed_all(10)
        np.random.seed(10)
        random.seed(10)
        # build the model and initialize
        print('loading model...')
        self.model = EdgeConnect(config)
        self.model.load()
        print('done.')
    
    def fill(self, img):
        outputs = self.model.test(np.array(img))
        output_image = outputs[0].cpu().numpy()
        print(output_image.dtype)
        return np.uint8(output_image)
    
    def load_config(self):

        config_path = os.path.join('./celeba', 'config.yml')

        # copy config template if does't exist
        if not os.path.exists(config_path):
            copyfile('./config.yml.example', config_path)

        # load config file
        config = Config(config_path)


        config.MODE = 2
        config.MODEL= 2
        config.INPUT_SIZE = 0
        config.MASK = 2
        
        
        config.TEST_FLIST = 'examples/celeba/images/celeba_02.png'
        #config.TEST_MASK_FLIST = 'examples/celeba/masks/celeba_02.png'
        config.RESULTS = 'output'

        return config