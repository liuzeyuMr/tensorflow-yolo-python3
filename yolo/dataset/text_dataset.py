from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import cv2
import numpy as np
from queue import Queue
from threading import Thread

from yolo.dataset.dataset import DataSet 

class TextDataSet(DataSet):
  """TextDataSet
  process text input file dataset 
  text file format:
    image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
  """

  def __init__(self, common_params, dataset_params):
    """
    Args:
      common_params: A dict
      dataset_params: A dict
    """
    #process params
    self.data_path = str(dataset_params['path'])
    self.width = int(common_params['image_size']) #448
    self.height = int(common_params['image_size'])#448
    self.batch_size = int(common_params['batch_size'])#16
    self.thread_num = int(dataset_params['thread_num'])#5
    self.max_objects = int(common_params['max_objects_per_image'])#20

    #record and image_label queue
    self.record_queue = Queue(maxsize=10000)
    self.image_label_queue = Queue(maxsize=512)

    self.record_list = []  

    # filling the record_list
    input_file = open(self.data_path, 'r')
    print('path',self.data_path)
    # print(input_file)
    for line in input_file:
      line = line.strip()
      # print(type(line))
      ss = line.split(' ') #按空格分开 保存为list
      # print(ss)
      ss[1:] = [float(num) for num in ss[1:]] #
      # print(ss)
      self.record_list.append(ss)

    self.record_point = 0
    self.record_number = len(self.record_list)

    self.num_batch_per_epoch = int(self.record_number / self.batch_size)

    t_record_producer = Thread(target=self.record_producer)
    t_record_producer.daemon = True 
    t_record_producer.start()

    for i in range(self.thread_num):
      t = Thread(target=self.record_customer)
      t.daemon = True
      t.start() 

  def record_producer(self):
    """record_queue's processor
    """
    while True:
      if self.record_point % self.record_number == 0:
        random.shuffle(self.record_list) #打乱
        self.record_point = 0
      self.record_queue.put(self.record_list[self.record_point])
      self.record_point += 1

  def record_process(self, record):
    """record process 
    Args: record 
    Returns:
      image: 3-D ndarray
      labels: 2-D list [self.max_objects, 5] (xcenter, ycenter, w, h, class_num)
      object_num:  total object number  int 
    """
    image = cv2.imread(record[0])
    #print('inmage',np.shape(image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print("111",image.shape)
    h = image.shape[0]
    w = image.shape[1]

    width_rate = self.width * 1.0 / w 
    height_rate = self.height * 1.0 / h 

    image = cv2.resize(image, (self.height, self.width))
    # print(image.shape)

    labels = [[0, 0, 0, 0, 0]] * self.max_objects
    i = 1
    object_num = 0
    while i < len(record):
      # print(len(record))
      xmin = record[i]   #对应真实框的值
      ymin = record[i + 1]
      xmax = record[i + 2]
      ymax = record[i + 3]
      class_num = record[i + 4] #所对应框的类别

      xcenter = (xmin + xmax) * 1.0 / 2 * width_rate #转换为中心点
      ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

      box_w = (xmax - xmin) * width_rate #转换为长和宽
      box_h = (ymax - ymin) * height_rate

      labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num] #对应图片的所有标签，包含框的中心点，长和宽，类别
      object_num += 1
      i += 5
      if object_num >= self.max_objects: #一共就只有20个类，如果图片对应超过了20类。。直接GG
        break
    return [image, labels, object_num] #返回图片 标签 图片包含的类别数

  def record_customer(self):
    """record queue's customer 
    """
    while True:
      item = self.record_queue.get()
      out = self.record_process(item)
      self.image_label_queue.put(out)#获得图片、标签和类别数

  def batch(self):
    """get batch
    Returns:
      images: 4-D ndarray [batch_size, height, width, 3]
      labels: 3-D ndarray [batch_size, max_objects, 5]
      objects_num: 1-D ndarray [batch_size]
    """
    images = []
    labels = []
    objects_num = []
    for i in range(self.batch_size):
      image, label, object_num = self.image_label_queue.get()
      images.append(image)
      labels.append(label)
      objects_num.append(object_num)
    images = np.asarray(images, dtype=np.float32)
    images = images/255 * 2 - 1
    labels = np.asarray(labels, dtype=np.float32)
    objects_num = np.asarray(objects_num, dtype=np.int32)
    return images, labels, objects_num