import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
from keras import utils as np_utils
import tensorflow as tf
from tensorflow.keras.optimizers import SGD


import sys
sys.path.append("..")
from utils import *
from funcs import *

from resnet import model_resnet
from DCASE_training_functions import *

#from tensorflow import ConfigProto
#from tensorflow import InteractiveSession

#tf.test.is_gpu_available()
print(tf.__version__)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


# Please put your csv file for train and validation here.
# If you dont generate the extra augmented data, please use 
# ../evaluation_setup/fold1_train.csv and delete the aug_csv part
train_csv = '/content/drive/MyDrive/dcase/evaluation_setup/fold1_train.csv'
val_csv = '/content/drive/MyDrive/dcase/evaluation_setup/fold1_evaluate.csv'
#aug_csv = 'evaluation_setup/fold1_train_a_2003.csv'

feat_path = '/content/drive/MyDrive/dcase/features/logmel64_scaled'
#aug_path = 'features/logmel128_reverb_scaled/'



experiments = '/content/drive/MyDrive/dcase/exp_2020_resnet_scaled_specaugment_timefreqmask_speccorr_together_nowd_alldata/'

if not os.path.exists(experiments):
    os.makedirs(experiments)


#train_aug_csv = generate_train_aug_csv(train_csv, aug_csv, feat_path, aug_path, experiments)
train_aug_csv = train_csv

num_audio_channels = 1
num_freq_bin = 64
num_classes = 10
max_lr = 0.1
batch_size = 64
num_epochs = 126
mixup_alpha = 0.4
crop_length = 400
sample_num = len(open(train_aug_csv, 'r').readlines()) - 1

# compute delta and delta delta for validation data
data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
data_deltas_val = deltas(data_val)
data_deltas_deltas_val = deltas(data_deltas_val)
data_val = np.concatenate((data_val[:,:,4:-4,:],data_deltas_val[:,:,2:-2,:],data_deltas_deltas_val),axis=-1)
y_val = keras.utils.np_utils.to_categorical(y_val, num_classes)

model = model_resnet(num_classes,
                     input_shape=[num_freq_bin,None,3*num_audio_channels], 
                     num_filters=48,
                     wd=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer =SGD(lr=max_lr,decay=1e-6, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.summary()

lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0]) 

save_path = experiments + "model-{epoch:02d}-{val_accuracy:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks = [lr_scheduler, checkpoint]

# Due to the memory limitation, in the training stage we split the training data
train_data_generator = Generator_withdelta_splitted(feat_path, train_aug_csv, num_freq_bin, 
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length, splitted_num=4)()

history = model.fit(train_data_generator,
                              validation_data=(data_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size = 100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              ) 
#history = model.fit_generator(train_data_generator,
      #                        validation_data=(data_val, y_val),
       #                       epochs=num_epochs,
       #                       verbose=1,
       #                       workers=4,
          #                    max_queue_size = 100,
         #                     callbacks=callbacks,
          #                    steps_per_epoch=np.ceil(sample_num/batch_size)
          #                    )
