from pprint import pprint

class Config:
    # data
    voc_data_dir = './data/VOCdevkit/VOC2012/'
    min_size = 448       # image resize
    max_size = 448       # image resize
    num_workers = 8      #8
    test_num_workers = 8 #8
    BATCH_SIZE = 1

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-4

    # visualization
    env = 'point-linking'  # visdom env
    port = 2333
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    #pretrained_model = 'vgg16'
    pretrained_model = 'inceptionresnetv2'
    # training
    epoch = 14


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    use_RMSprop = True    
    # use_RMSprop = False

    test_num = 2
    # model
    # load_path = None
    load_path ='./checkpoints/2206_0.05w_0.0001_no_sigmoid06172150_0.05521642910775096.pth'

    use_pretrain = True # use caffe pretrained model instead of torchvision
    #caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'
    pretrain_path = './checkpoints/inceptionresnetv2.pth'
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('=========config==========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


option = Config()
