from __future__ import division
import os, sys
sys.path.append(os.path.abspath("./"))

import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data_factory.dataset_factory import ImageCollector

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

        self.real_A = self.real_data[:, :, :, :self.A_dim]
        self.real_B = self.real_data[:, :, :, self.A_dim:self.A_dim + self.B_dim]

        self.fake_B = self.generator(self.real_A, self.options, self.B_dim, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, self.A_dim, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, self.A_dim, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, self.B_dim, True, name="generatorA2B")

        self.DB_fake1 = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB1", layer=2)
        self.DA_fake1 = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA1", layer=2)
        self.DB_fake2 = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB2", layer=3)
        self.DA_fake2 = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA2", layer=3)
        self.DB_fake3 = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB3", layer=4)
        self.DA_fake3 = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA3", layer=4)


        self.g_loss_a2b = self.criterionGAN(self.DB_fake1, tf.ones_like(self.DB_fake1)) \
            + self.criterionGAN(self.DB_fake2, tf.ones_like(self.DB_fake2)) \
            + self.criterionGAN(self.DB_fake3, tf.ones_like(self.DB_fake3)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake1, tf.ones_like(self.DA_fake1)) \
            + self.criterionGAN(self.DA_fake2, tf.ones_like(self.DA_fake2)) \
            + self.criterionGAN(self.DA_fake3, tf.ones_like(self.DA_fake3)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake1, tf.ones_like(self.DA_fake1)) \
            + self.criterionGAN(self.DA_fake2, tf.ones_like(self.DA_fake2)) \
            + self.criterionGAN(self.DA_fake3, tf.ones_like(self.DA_fake3)) \
            + self.criterionGAN(self.DB_fake1, tf.ones_like(self.DB_fake1)) \
            + self.criterionGAN(self.DA_fake2, tf.ones_like(self.DA_fake2)) \
            + self.criterionGAN(self.DA_fake3, tf.ones_like(self.DA_fake3)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.A_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.B_dim], name='fake_B_sample')
        self.DB_real1 = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB1", layer=2)
        self.DA_real1 = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA1", layer=2)
        self.DB_real2 = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB2", layer=3)
        self.DA_real2 = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA2", layer=3)
        self.DB_real3 = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB3", layer=4)
        self.DA_real3 = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA3", layer=4)
        self.DB_fake_sample1 = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB1", layer=2)
        self.DA_fake_sample1 = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA1", layer=2)
        self.DB_fake_sample2 = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB2", layer=3)
        self.DA_fake_sample2 = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA2", layer=3)
        self.DB_fake_sample3 = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB3", layer=4)
        self.DA_fake_sample3 = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA3", layer=4)

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

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.A_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.B_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, self.B_dim, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, self.A_dim, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        self.datasetA = ImageCollector("C:\\Users\\incorl\\Desktop\\GraspGAN\\domain_adaptation\\pixel_domain_adaptation\\simul_dataset", 1, 100, self.batch_size, bCollectSeg=True)  # Simul data A
        self.datasetB = ImageCollector("C:\\Users\\incorl\\Desktop\\GraspGAN\\domain_adaptation\\pixel_domain_adaptation\\real_dataset", 1, 100, self.batch_size)  # Real data B
        
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
                dataA = np.concatenate((dataA[1], dataA[2]), axis=3)
                batch_files = list(zip(dataA, dataB[1]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
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
        dataA = np.concatenate((dataA[1], dataA[2]), axis=3)
        batch_files = list(zip(dataA, dataB[1]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )


        images = np.split(sample_images, [3], axis=3)

        concat_B = np.concatenate((images[0], fake_B), axis=2)
        concat_A = np.concatenate((images[1], fake_A), axis=2)
        save_images(concat_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(concat_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

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
