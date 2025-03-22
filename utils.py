"""
    Some handy functions for pytroch model training ...
"""
import torch
import logging
import numpy as np
import scipy.sparse as sp
import copy
from transformers import BertTokenizer, BertModel, CLIPModel, CLIPProcessor
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H-%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = sp.spdiags(idf, 0, col, col)
    return tf * idf


def process_and_save_image_features_vit(image_folder, output_pt_path):
    # 加载 CLIP ViT 模型和预处理器
    vit_model = CLIPModel.from_pretrained("../CLIP-vit-large-patch14").to('cuda:0').eval()
    vit_processor = CLIPProcessor.from_pretrained("../CLIP-vit-large-patch14")

    features_dict = {}

    # 遍历文件夹中的所有图片
    for img_name in tqdm(os.listdir(image_folder), desc="Processing images"):
        img_path = os.path.join(image_folder, img_name)

        try:
            # 加载图片并转换为 RGB 格式
            image = Image.open(img_path).convert("RGB")

            # 预处理图片
            inputs = vit_processor(images=image, return_tensors="pt", padding=True).to('cuda:0')

            # 提取特征
            with torch.no_grad():
                outputs = vit_model.get_image_features(**inputs)
                features = outputs.cpu().squeeze(0)  # 转换到 CPU 并移除多余维度
            # 使用图片名称（去掉扩展名）作为 ID
            img_id = int(os.path.splitext(img_name)[0])
            # print(img_id)
            features_dict[img_id] = features

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            continue

    # 保存为 .pt 文件
    torch.save(features_dict, output_pt_path)
    print(f"Features saved to {output_pt_path}")

def process_and_save_text_features_bert(text_file, output_pt_path):
    # 加载 BERT 模型和预处理器
    bert_model = BertModel.from_pretrained("../bert-base").to('cuda:0').eval()
    bert_tokenizer = BertTokenizer.from_pretrained("../bert-base")

    features_dict = {}
    # 读取文本数据
    text_data = pd.read_csv(text_file,header=None,usecols=[0,2])
    # 遍历文本数据
    for i in tqdm(range(len(text_data)), desc="Processing text"):
        text = text_data.iloc[i,1]
        # 预处理文本
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True).to('cuda:0')
        # 提取特征
        with torch.no_grad():
            outputs = bert_model(**inputs)
            features = outputs.last_hidden_state[:,0,:].cpu().squeeze(0)  # 转换到 CPU 并移除多余维度
        # 使用文本 ID 作为 ID
        text_id = int(text_data.iloc[i,0])
        # print(text_id)
        features_dict[text_id] = features

    # 保存为 .pt 文件
    torch.save(features_dict, output_pt_path)
    print(f"Features saved to {output_pt_path}")


def load_data(data_file):
    id_data = pd.read_csv(data_file+'/DY_pair_mini.csv', header=None, usecols=[0,1])

    user_ids = list(set(id_data.iloc[:,1]))
    item_ids = list(set(id_data.iloc[:,0]))

    item_img_dict = torch.load("../data/DY/DY_vit.pt")
    item_text_dict = torch.load("../data/DY/DY_bert.pt")

    item_img_features = {int(keys): v for keys, v in item_img_dict.items()}
    item_text_features = {int(keys): v for keys, v in item_text_dict.items()}

    # 读取训练数据
    train_id_data = pd.read_csv(data_file + '/train_mini.csv')

    # 创建特征列表
    train_img_features = []
    train_text_features = []

    # 逐个获取每个物品的特征
    for iid in train_id_data['iid']:
        train_img_features.append(item_img_features[iid])
        train_text_features.append(item_text_features[iid])

    # 转换为张量
    train_img_features = torch.stack(train_img_features)
    train_text_features = torch.stack(train_text_features)

    # 读取测试数据
    test_id_data = pd.read_csv(data_file + '/test_mini.csv')

    # 创建特征列表
    test_img_features = []
    test_text_features = []

    # 逐个获取每个物品的特征
    for iid in test_id_data['iid']:
        test_img_features.append(item_img_features[iid])
        test_text_features.append(item_text_features[iid])

    # 转换为张量
    test_img_features = torch.stack(test_img_features)
    test_text_features = torch.stack(test_text_features)


    data_dict = {
        'train_data': train_id_data,
        'train_img_features': train_img_features,
        'train_text_features': train_text_features,
        'test_data': test_id_data,
        'test_img_features': test_img_features,
        'test_text_features': test_text_features,
        'user_ids': user_ids,
        'item_ids': item_ids,
        'item_img_features': item_img_features,  # 添加完整的字典以便后续使用
        'item_text_features': item_text_features
    }
    return data_dict


def negative_sampling(train_data, num_negatives):
    """sample negative instances for training, refer to Heater."""
    # warm items in training set.
    item_warm = np.unique(train_data['iid'].values)
    # arrange the training data with form {user_1: [[user_1], [user_1_item], [user_1_rating]],...}.
    train_dict = {}
    single_user, user_item, user_rating = [], [], []
    grouped_train_data = train_data.groupby('uid')
    for userId, user_train_data in grouped_train_data:
        temp = copy.deepcopy(item_warm)
        for row in user_train_data.itertuples():
            single_user.append(int(row.uid))
            user_item.append(int(row.iid))
            user_rating.append(float(1))
            temp = np.delete(temp, np.where(temp == row.iid))
            for i in range(num_negatives):
                single_user.append(int(row.uid))
                negative_item = np.random.choice(temp)
                user_item.append(int(negative_item))
                user_rating.append(float(0))
                temp = np.delete(temp, np.where(temp == negative_item))
        train_dict[userId] = [single_user, user_item, user_rating]
        single_user = []
        user_item = []
        user_rating = []
    return train_dict


def compute_metrics_with_mapping(evaluate_data, user_item_preds, k_list, is_test=True):
    """
    计算推荐评估指标，正确处理物品ID映射

    Args:
        evaluate_data: 包含uid和iid的DataFrame
        user_item_preds: 字典 {user_id: {item_id: score}}
        k_list: 评估使用的k值列表
        is_test: 是否为测试集评估

    Returns:
        recalls, precisions, ndcgs: 不同k值下的评估指标
    """
    # 按用户分组计算评估指标
    recalls = [0.0 for _ in range(len(k_list))]
    precisions = [0.0 for _ in range(len(k_list))]
    ndcgs = [0.0 for _ in range(len(k_list))]

    # 获取所有用户
    users = evaluate_data['uid'].unique()
    num_users = len(users)

    for user in users:
        # 获取用户的真实交互物品
        user_data = evaluate_data[evaluate_data['uid'] == user]
        gt_items = set(user_data['iid'].values)

        if user not in user_item_preds:
            continue

        # 获取用户的预测分数
        user_pred = user_item_preds[user]

        # 排除已知的训练集物品（如果需要）
        # pred_items = {k: v for k, v in user_pred.items() if k not in train_items}

        # 按分数排序的物品
        pred_items_list = sorted(user_pred.items(), key=lambda x: x[1], reverse=True)
        pred_items = [item[0] for item in pred_items_list]

        # 计算不同k值下的指标
        for i, k in enumerate(k_list):
            # 取前k个推荐结果
            topk_items = pred_items[:k]
            # 计算召回率 (命中数/真实物品总数)
            recalls[i] += len(set(topk_items) & gt_items) / len(gt_items)
            # 计算精确率 (命中数/推荐数)
            precisions[i] += len(set(topk_items) & gt_items) / len(topk_items)

            # 计算NDCG
            idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(gt_items), k))])
            dcg = 0.0
            for idx, item in enumerate(topk_items):
                if item in gt_items:
                    dcg += 1.0 / np.log2(idx + 2)
            ndcgs[i] += dcg / idcg if idcg > 0 else 0

    # 计算平均值
    for i in range(len(k_list)):
        recalls[i] /= num_users
        precisions[i] /= num_users
        ndcgs[i] /= num_users

    return recalls, precisions, ndcgs

# def compute_regularization(model, parameter_label):
#     reg_fn = torch.nn.MSELoss(reduction='mean')
#     for name, param in model.named_parameters():
#         if name == 'embedding_item.weight':
#             reg_loss = reg_fn(param, parameter_label)
#             return reg_loss
#定义损失函数：使用 torch.nn.MSELoss(reduction='mean') 定义一个均方误差损失函数 reg_fn。
# 遍历模型参数：通过 model.named_parameters() 遍历模型的所有命名参数，每次迭代得到一个参数的名称 name 和参数本身 param。
# 检查参数名称：判断参数名称是否为 embedding_item.weight。
# 计算并返回损失：如果参数名称匹配，使用 reg_fn 计算该参数与 parameter_label 之间的均方误差损失，并将结果存储在 reg_loss 中，然后立即返回该损失值。

# if __name__ == '__main__':
#     data_dict = load_data('../data/Bili_Food')
#     print(len(data_dict['train_item_ids_map']))
#     print(len(data_dict['test_item_ids_map']))
#     print(len(data_dict['vali_item_ids_map']))
#
