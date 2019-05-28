import sys
# print(sys.path)
sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet  #导入网络对象
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:20] #预测
  # print(type(p_classes))
  C = predicts[0, :, :, 20:22] #置信度
  print("@@@@@@@",C[3,3,0])
  print("######",np.max(C))
  confidence=np.max(C)
  print(np.unravel_index(np.argmax(C), C.shape))
  coordinate = predicts[0, :, :, 22:] #边框
  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))
  print(type(C))
  P = C * p_classes  #这里是numpy的数组相乘，p_classes维度低于P会自动复制进行扩张(7, 7, 2, 20))  边界框类别置信度
  print(np.max(P))

  index = np.argmax(P)


  index = np.unravel_index(index, P.shape) #(3, 3, 0, 7) 表示在第3*3个特征点第一个预测框检测到了第七个类别 猫
  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :] #获得对应的特征点的第一个预测框的坐标信息
  print(max_coordinate)

  xcenter = max_coordinate[0] #表示代表的图片区域的中心点
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  # print(xcenter)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448 #分别表示边框的长和高
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h
 #输出左上角和右下角坐标
  return xmin, ymin, xmax, ymax, class_num,confidence


common_params = {'image_size': 448, 'num_classes': 20, 
                'batch_size':1}

net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005} #网络结构参数

net = YoloTinyNet(common_params, net_params, test=True) #定义网络 传入字典 获得网络类对象

image = tf.placeholder(tf.float32, (1, 448, 448, 3)) # 图片大小
predicts = net.inference(image)
# print(predicts.shape)
sess = tf.Session()

np_img = cv2.imread('dog.jpg')
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) #颜色通道转换


np_img = np_img.astype(np.float32)

np_img = np_img / 255.0 * 2 - 1 #预处理[-1,1]
np_img = np.reshape(np_img, (1, 448, 448, 3))

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt') #加载测试模型

np_predict = sess.run(predicts, feed_dict={image: np_img})

xmin, ymin, xmax, ymax, class_num,confidence = process_predicts(np_predict)#
class_name = classes_name[class_num]
cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255)) #
label='{},{}'.format(class_name,confidence)
cv2.putText(resized_img, label, (int(xmin), int(ymax)), 1, 1.0, (0, 0, 255))
cv2.imwrite('dog_out.jpg', resized_img)
sess.close()
