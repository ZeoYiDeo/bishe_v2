import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from mlp import MLPEngine
from utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='fedcs')
parser.add_argument('--clients_sample_ratio', type=float, default=1)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--server_epoch', type=int, default=1)
# parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--reg', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr_client', type=float, default=0.05)
parser.add_argument('--lr_server', type=float, default=0.005)
parser.add_argument('--alpha', type=float, default=0.7)
# parser.add_argument('--total_steps', type=int, default=1000)
parser.add_argument('--dataset', type=str, default='DY')
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_items_train', type=int)
parser.add_argument('--num_items_vali', type=int)
parser.add_argument('--num_items_test', type=int)
parser.add_argument('--content_dim', type=int)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--num_negative', type=int, default=3)
# 修改网络层配置
# parser.add_argument('--server_model_layers', type=str, default='1536,768,384')
parser.add_argument('--client_model_layers', type=str, default='64')  # 从latent_dim开始
parser.add_argument('--recall_k', type=str, default='20,50,100')
parser.add_argument('--l2_regularization', type=float, default=1e-5)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=1)

args = parser.parse_args()

# Model.
config = vars(args)

if len(config['recall_k']) > 1:
    config['recall_k'] = [int(item) for item in config['recall_k'].split(',')]
else:
    config['recall_k'] = [int(config['recall_k'])]

if len(config['client_model_layers']) > 1:
    config['client_model_layers'] = [int(item) for item in config['client_model_layers'].split(',')]
else:
    config['client_model_layers'] = int(config['client_model_layers'])
if config['dataset'] == 'Bili_Food':
    config.update({
        'num_users': 1926,
        # 'num_items_train': 6898,
        # 'num_items_vali': 2378,
        # 'num_items_test': 2378,
        # 'content_dim': 768,
    })
elif config['dataset'] == 'DY':
    config.update({
        'num_users': 2328,
        'num_items_train': 15584,
        'num_items_test': 3156,
        # 'content_dim': 300,
    })
else:
    pass

engine = MLPEngine(config)
# 设置物品特征（假设您已经加载了这些特征）


# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load Data
dataset_dir = "../data/" + config['dataset']
data_dict = load_data(dataset_dir)
# train, validation and test data with (uid, iid) dataframe format.
train_data = data_dict['train_data']
test_data = data_dict['test_data']
# train, validation, test cold-start item information, including item raw feature and reindex item id dict {ori_id: reindex_id}
train_item_img_features = data_dict['train_img_features']
train_item_text_features = data_dict['train_text_features']
user_ids = data_dict['user_ids']

test_item_img_features = data_dict['test_img_features']
test_item_text_features = data_dict['test_text_features']

# print(train_item_img_features.shape, train_item_text_features.shape)

engine.item_cv_features = train_item_img_features  # 从某处加载的CV特征
engine.item_text_features = train_item_text_features  # 从某处加载的文本特征


test_recalls = []
test_precisions = []
test_ndcgs = []
best_recall = 0
final_test_round = 0
for round in range(config['num_round']):
    # # break
    logging.info('-' * 80)
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round))

    all_train_data = negative_sampling(train_data, config['num_negative'])
    logging.info('-' * 80)
    logging.info('Training phase!')
    engine.fed_train_a_round(user_ids, all_train_data, round)

    logging.info('-' * 80)
    logging.info('Testing phase!')
    test_recall, test_precision, test_ndcg = engine.fed_evaluate(test_data, test_item_img_features, test_item_text_features,is_test=True)
    logging.info('Recall@{} = {:.6f}, Recall@{} = {:.6f}, Recall@{} = {:.6f}'.format(
        config['recall_k'][0], test_recall[0],
        config['recall_k'][1], test_recall[1],
        config['recall_k'][2], test_recall[2]
    ))
    logging.info('Precision@{} = {:.6f}, Precision@{} = {:.6f}, Precision@{} = {:.6f}'.format(
        config['recall_k'][0], test_precision[0],
        config['recall_k'][1], test_precision[1],
        config['recall_k'][2], test_precision[2]
    ))
    logging.info('NDCG@{} = {:.6f}, NDCG@{} = {:.6f}, NDCG@{} = {:.6f}'.format(
        config['recall_k'][0], test_ndcg[0],
        config['recall_k'][1], test_ndcg[1],
        config['recall_k'][2], test_ndcg[2]
    ))
    test_recalls.append(test_recall)
    test_precisions.append(test_precision)
    test_ndcgs.append(test_ndcg)

    # logging.info('-' * 80)
    # logging.info('Validating phase!')
    # vali_recall, vali_precision, vali_ndcg = engine.fed_evaluate(vali_data, vali_item_img_features, vali_item_text_features, is_test=False)
    # logging.info('Recall@{} = {:.6f}, Recall@{} = {:.6f}, Recall@{} = {:.6f}'.format(
    #     config['recall_k'][0], vali_recall[0],
    #     config['recall_k'][1], vali_recall[1],
    #     config['recall_k'][2], vali_recall[2]
    # ))
    # logging.info('Precision@{} = {:.6f}, Precision@{} = {:.6f}, Precision@{} = {:.6f}'.format(
    #     config['recall_k'][0], vali_precision[0],
    #     config['recall_k'][1], vali_precision[1],
    #     config['recall_k'][2], vali_precision[2]
    # ))
    # logging.info('NDCG@{} = {:.6f}, NDCG@{} = {:.6f}, NDCG@{} = {:.6f}'.format(
    #     config['recall_k'][0], vali_ndcg[0],
    #     config['recall_k'][1], vali_ndcg[1],
    #     config['recall_k'][2], vali_ndcg[2]
    # ))
    # logging.info('')
    # vali_recalls.append(vali_recall)
    # vali_precisions.append(vali_precision)
    # vali_ndcgs.append(vali_ndcg)

    if np.sum(test_recall) >= np.sum(best_recall):
        best_recall = test_recall
        final_test_round = round

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
str = current_time + '-' + 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr_client: ' + str(config['lr_client']) \
      + '-'  + 'local_epoch: ' + str(config['local_epoch']) + '-' + \
      '-' + 'client_model_layers: ' + str(config['client_model_layers']) + '-' \
      'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'num_round: ' + str(config['num_round']) \
      + '-' + 'negatives: ' + str(config['num_negative']) + '-' + \
      'batch_size: ' + str(config['batch_size']) + '-' + 'Recall: ' + str(test_recalls[final_test_round]) + '-' \
      + 'Precision: ' + str(test_precisions[final_test_round]) + '-' + 'NDCG: ' + str(test_ndcgs[final_test_round]) + '-' \
      + 'best_round: ' + str(final_test_round) + '-' + 'recall_k: ' + str(config['recall_k']) + '-' + \
      'optimizer: ' + config['optimizer'] + '-' + 'l2_regularization: ' + str(config['l2_regularization'])
file_name = "sh_result/"+config['dataset']+".txt"
if not os.path.exists(file_name):
    # 创建空文件
    with open(file_name, 'w') as f:
        pass  # 创建空文件

# 正常追加写入内容
with open(file_name, 'a') as file:
    file.write(str + '\n')

logging.info('fedcs')
logging.info('recall_list: {}'.format(test_recalls))
logging.info('precision_list: {}'.format(test_precisions))
logging.info('ndcg_list: {}'.format(test_ndcgs))
logging.info('clients_sample_ratio: {}, bz: {}, lr_client: {}, local_epoch: {},client_model_layers: {}, recall_k: {}, dataset: {}, '
             'factor: {}, negatives: {},'.format(config['clients_sample_ratio'],
                                                         config['batch_size'], config['lr_client'],
                                                         config['local_epoch'],
                                                         config['client_model_layers'],
                                                         config['recall_k'], config['dataset'], config['latent_dim'],
                                                         config['num_negative']))
logging.info('Best test recall: {}, precision: {}, ndcg: {} at round {}'.format(test_recalls[final_test_round],
                                                                                test_precisions[final_test_round],
                                                                                test_ndcgs[final_test_round],
                                                                                final_test_round))
