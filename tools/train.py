import sys
from optparse import OptionParser

sys.path.append('./')
sys.path.append('../')
import yolo
from yolo.utils.process_config import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename",default='E:/objectdetection/tensorflow-yolo-python3/tensorflow-yolo-python3/conf/train.cfg')
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
  # print(conf_file)
else:
  print('please specify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = yolo.dataset.text_dataset.TextDataSet(common_params, dataset_params) #yolo.dataset.text_dataset.TextDataSet（common_params, dataset_params） 获得数据集
net = yolo.net.yolo_tiny_net.YoloTinyNet(common_params, net_params) # 获得网络结构对象
solver = yolo.solver.yolo_solver.YoloSolver(dataset, net, common_params, solver_params)
solver.solve()