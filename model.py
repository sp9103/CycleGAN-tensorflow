from __future__ import division
import os, sys
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.realpath(os.path.dirname(__file__)))

import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data_factory.dataset_factory import ImageCollector
import cv2

from module import *
from utils import *

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.A_dim = args.A_nc
        self.B_dim = args.B_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.A_dim + self.B_dim],
                                        name='real_A_and_B_images')
        self.seg_data = tf.placeholder(tf.float32,
                                       [None, self.image_size, self.image_size, 1],
                                       name='segment_images')

        self.real_A = self.real_data[:, :, :, :self.A_dim]
        self.real_B = self.real_data[:, :, :, self.A_dim:self.A_dim + self.B_dim]

        self.fake_B = self.generator(self.real_A, self.options, self.B_dim, False, name="generatorA2B")     #simulator에서 출발하는 cycle
        self.fake_A_ = self.generator(self.fake_B, self.options, self.A_dim, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, self.A_dim, True, name="generatorB2A")      #real에서 출발하는 cycle - self.fake_segA
        self.fake_B_ = self.generator(self.fake_A, self.options, self.B_dim, True, name="generatorA2B")    # - self.fake_segB_

        self.real_seg_A = seg_generator_resnet(self.real_A, self.options, 1, False, name="segnetA")         # 학습할 수 있는 것 1 -> segnet을 학습할 때 사용함
        self.fake_seg_B = seg_generator_resnet(self.fake_B, self.options, 1, False, name="segnetB")          # 학습할 수 있는 것 2 ? (일단 제외) -> 이게 되면 A, B 도메인 둘다 sematic seg가 가능해진다.
        self.fake_seg_A_ = seg_generator_resnet(self.fake_A_, self.options, 1, True, name="segnetA")        # 학습할 수 있는 것 3 -> g step에서 사용함.
        self.real_seg_B = seg_generator_resnet(self.real_B, self.options, 1, True, name="segnetB")
        self.fake_seg_A = seg_generator_resnet(self.fake_A, self.options, 1, True, name="segnetA")
        self.fake_seg_B_ = seg_generator_resnet(self.fake_B_, self.options, 1, True, name="segnetB")

        self.DB_fake1 = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB1", layers=1)
        self.DA_fake1 = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA1", layers=1)
        self.DB_fake2 = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB2", layers=2)
        self.DA_fake2 = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA2", layers=2)
        self.DB_fake3 = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB3", layers=3)
        self.DA_fake3 = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA3", layers=3)

        self.g_loss_a2b = self.criterionGAN(self.DB_fake1, tf.ones_like(self.DB_fake1)) \
            + self.criterionGAN(self.DB_fake2, tf.ones_like(self.DB_fake2)) \
            + self.criterionGAN(self.DB_fake3, tf.ones_like(self.DB_fake3)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + self.L1_lambda * abs_criterion(self.real_seg_A, self.fake_seg_A_) \
            + self.L1_lambda * abs_criterion(self.real_seg_B, self.fake_seg_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake1, tf.ones_like(self.DA_fake1)) \
            + self.criterionGAN(self.DA_fake2, tf.ones_like(self.DA_fake2)) \
            + self.criterionGAN(self.DA_fake3, tf.ones_like(self.DA_fake3)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + self.L1_lambda * abs_criterion(self.real_seg_A, self.fake_seg_A_) \
            + self.L1_lambda * abs_criterion(self.real_seg_B, self.fake_seg_B_)
        self.g_loss = self.criterionGAN(self.DA_fake1, tf.ones_like(self.DA_fake1)) \
            + self.criterionGAN(self.DA_fake2, tf.ones_like(self.DA_fake2)) \
            + self.criterionGAN(self.DA_fake3, tf.ones_like(self.DA_fake3)) \
            + self.criterionGAN(self.DB_fake1, tf.ones_like(self.DB_fake1)) \
            + self.criterionGAN(self.DB_fake2, tf.ones_like(self.DB_fake2)) \
            + self.criterionGAN(self.DB_fake3, tf.ones_like(self.DB_fake3)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + self.L1_lambda * abs_criterion(self.real_seg_A, self.fake_seg_A_) \
            + self.L1_lambda * abs_criterion(self.real_seg_B, self.fake_seg_B_) \
            + self.L1_lambda * abs_criterion(self.fake_seg_A, self.real_seg_B)

        # 실험1. label이 확실한 simulator data만 가지고 학습한다.
        self.seg_loss = self.L1_lambda * abs_criterion(self.real_seg_A, self.seg_data) \
            + self.L1_lambda * abs_criterion(self.fake_seg_B, self.seg_data)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.A_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.B_dim], name='fake_B_sample')
        self.DB_real1 = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB1", layers=1)
        self.DA_real1 = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA1", layers=1)
        self.DB_real2 = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB2", layers=2)
        self.DA_real2 = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA2", layers=2)
        self.DB_real3 = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB3", layers=3)
        self.DA_real3 = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA3", layers=3)
        self.DB_fake_sample1 = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB1", layers=1)
        self.DA_fake_sample1 = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA1", layers=1)
        self.DB_fake_sample2 = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB2", layers=2)
        self.DA_fake_sample2 = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA2", layers=2)
        self.DB_fake_sample3 = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB3", layers=3)
        self.DA_fake_sample3 = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA3", layers=3)

        self.db_loss_real = self.criterionGAN(self.DB_real1, tf.ones_like(self.DB_real1)) \
            + self.criterionGAN(self.DB_real2, tf.ones_like(self.DB_real2)) \
            + self.criterionGAN(self.DB_real3, tf.ones_like(self.DB_real3))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample1, tf.zeros_like(self.DB_fake_sample1)) \
            + self.criterionGAN(self.DB_fake_sample2, tf.zeros_like(self.DB_fake_sample2)) \
            + self.criterionGAN(self.DB_fake_sample3, tf.zeros_like(self.DB_fake_sample3))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real1, tf.ones_like(self.DA_real1)) \
            + self.criterionGAN(self.DA_real2, tf.ones_like(self.DA_real2)) \
            + self.criterionGAN(self.DA_real3, tf.ones_like(self.DA_real3))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample1, tf.zeros_like(self.DA_fake_sample1)) \
            + self.criterionGAN(self.DA_fake_sample2, tf.zeros_like(self.DA_fake_sample2)) \
            + self.criterionGAN(self.DA_fake_sample3, tf.zeros_like(self.DA_fake_sample3))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )
        self.s_sum = tf.summary.scalar("seg_loss", self.seg_loss)

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.A_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.B_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, self.B_dim, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, self.A_dim, True, name="generatorB2A")
        self.test_seg_A = seg_generator_resnet(self.test_A, self.options, 1, True, name="segnetA")
        self.test_seg_B = seg_generator_resnet(self.test_B, self.options, 1, True, name="segnetB")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.s_vars = [var for var in t_vars if 'segnet' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.seg_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.seg_loss, var_list=self.s_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        self.datasetB = ImageCollector("C:\\Users\\incorl\\Desktop\\GraspGAN\\real_dataset_v2", 1, 100, self.batch_size)  # Real data B
        self.datasetA = ImageCollector("C:\\Users\\incorl\\Desktop\\GraspGAN\\simul_dataset_v2", 1, 100, self.batch_size, bCollectSeg=True)  # Simul data A
        
        self.datasetA.StartLoadData()
        self.datasetB.StartLoadData()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            batch_idxs = min(min(self.datasetA.getDataCnt(), self.datasetB.getDataCnt()), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                dataA = self.datasetA.getLoadedData()
                dataB = self.datasetB.getLoadedData()
                # dataA = np.concatenate((dataA[1], dataA[2]), axis=3)
                batch_files = list(zip(dataA[1], dataB[1]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size, is_testing=True) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                seg = cv2.resize(dataA[2][0], (args.fine_size, args.fine_size))
                seg = np.reshape(seg, (1, args.fine_size, args.fine_size, 1))

                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str, gen_seg_A = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum, self.fake_seg_A_],
                    feed_dict={self.real_data: batch_images, self.lr: lr, self.seg_data: seg})
                self.writer.add_summary(summary_str, counter)

                # Update Seg network
                _, summary_str = self.sess.run(
                    [self.seg_optim, self.s_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr, self.seg_data: seg})
                self.writer.add_summary(summary_str, counter)

                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = self.datasetA.getLoadedData()
        dataB = self.datasetB.getLoadedData()
        # dataA = np.concatenate((dataA[1], dataA[2]), axis=3)
        batch_files = list(zip(dataA[1], dataB[1]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)
        seg = cv2.resize(dataA[2][0], (256, 256))
        seg = np.reshape(seg, (1, 256, 256, 1))

        fake_A, fake_B, real_seg_A, fake_seg_A, fake_seg_B, \
        fake_seg_A_, fake_A_, fake_B_, real_seg_B, fake_seg_B_ = self.sess.run(
            [self.fake_A, self.fake_B, self.real_seg_A, self.fake_seg_A, self.fake_seg_B,
             self.fake_seg_A_, self.fake_A_, self.fake_B_, self.real_seg_B, self.fake_seg_B_],
            feed_dict={self.real_data: sample_images}
        )

        images = np.split(sample_images, [3], axis=3)

        real_seg_A = np.reshape(cv2.cvtColor(real_seg_A[0], cv2.COLOR_GRAY2RGB), (1, 256, 256, 3))
        real_seg_B = np.reshape(cv2.cvtColor(real_seg_B[0], cv2.COLOR_GRAY2RGB), (1, 256, 256, 3))
        fake_seg_A = np.reshape(cv2.cvtColor(fake_seg_A[0], cv2.COLOR_GRAY2RGB), (1, 256, 256, 3))
        fake_seg_B = np.reshape(cv2.cvtColor(fake_seg_B[0], cv2.COLOR_GRAY2RGB), (1, 256, 256, 3))
        fake_seg_A_ = np.reshape(cv2.cvtColor(fake_seg_A_[0], cv2.COLOR_GRAY2RGB), (1, 256, 256, 3))
        fake_seg_B_ = np.reshape(cv2.cvtColor(fake_seg_B_[0], cv2.COLOR_GRAY2RGB), (1, 256, 256, 3))

        # concat_B = np.concatenate((images[0], fake_B, real_seg_A), axis=2)
        # concat_A = np.concatenate((images[1], fake_A, fake_seg_A), axis=2)
        concat_cycleABA = np.concatenate((images[0], fake_B, fake_A_, real_seg_A, fake_seg_B, fake_seg_A_), axis=2)
        concat_cycleBAB = np.concatenate((images[1], fake_A, fake_B_, real_seg_B, fake_seg_A, fake_seg_B_), axis=2)
        # save_images(concat_A, [self.batch_size, 1],
        #             './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        # save_images(concat_B, [self.batch_size, 1],
        #             './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(concat_cycleBAB, [self.batch_size, 1],
                    './{}/cycleBAB_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(concat_cycleABA, [self.batch_size, 1],
                    './{}/cycleABA_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
