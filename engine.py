import torch
from utils import *
import numpy as np
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm



class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.client_model and self.server_model!
    """

    def __init__(self, config):
        self.config = config  # model configuration

        self.server_model_param = {}
        self.client_model_params = {}
        self.client_crit = torch.nn.BCELoss()
        # self.server_crit = torch.nn.MSELoss()

    def lambda_loss(self, prediction, label, lambda_val=0.05):
        """
        Compute the lambda loss for ranking optimization
        Args:
            prediction: model predictions (batch_size, 1)
            label: true labels (batch_size, 1)
            lambda_val: regularization parameter for balancing the loss

        Returns:
            total_loss: combined loss value
        """
        # 确保输入tensor格式正确
        prediction = prediction.view(-1)
        label = label.float().view(-1)

        # 计算基础的BCE损失
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction,
            label,
            reduction='none'
        )

        # 计算排序损失部分
        batch_size = prediction.size(0)
        pos_mask = (label == 1)
        neg_mask = (label == 0)

        if torch.sum(pos_mask) > 0 and torch.sum(neg_mask) > 0:
            pos_pred = prediction[pos_mask]
            neg_pred = prediction[neg_mask]

            # 计算正负样本对的差异
            diff_matrix = pos_pred.unsqueeze(1) - neg_pred.unsqueeze(0)  # shape: (num_pos, num_neg)

            # 使用sigmoid计算排序损失
            rank_loss = -torch.log(torch.sigmoid(diff_matrix) + 1e-8)
            rank_loss = torch.mean(rank_loss)

            # 组合BCE损失和排序损失
            total_loss = torch.mean(bce_loss) + lambda_val * rank_loss
        else:
            # 如果批次中没有正样本或负样本，则只使用BCE损失
            total_loss = torch.mean(bce_loss)



        return total_loss

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=64, shuffle=True)

    def fed_train_single_batch(self, model_client, batch, optimizer, round_id):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings,cv,ct = batch[0], batch[1], batch[2], batch[3], batch[4]
        ratings = ratings.float()
        ratings = ratings.cuda()
        # reg_item_embedding = copy.deepcopy(self.server_model_param['global_item_rep'])  #服务器的全局物品表示，用于计算后续的2正则化项

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            # reg_item_embedding = reg_item_embedding.cuda()
        #分别为客户端的优化器、用户嵌入的优化器、物品嵌入的优化器
        # update score function.
        model_client.train()
        optimizer.zero_grad()
        ratings_pred = model_client(cv,ct)

        loss = self.client_crit(ratings_pred.view(-1),ratings)
        # loss += self.config['reg'] * regularization_term
        # l2_reg = 0
        # for param in model_client.parameters():
        #     l2_reg += torch.norm(param)
        #
        # # 如果不是第一轮，添加知识蒸馏损失
        # if round_id > 0 and hasattr(self, 'prev_model_params'):
        #     kd_loss = 0
        #     # 计算当前预测与上一轮预测的KL散度
        #     with torch.no_grad():
        #         prev_pred = self.get_previous_prediction(batch)
        #     kd_loss = F.kl_div(
        #         F.log_softmax(ratings_pred, dim=0),
        #         F.softmax(prev_pred, dim=0)
        #     )
        #     loss += 0.1 * kd_loss  # 知识蒸馏权重
        #
        # loss += 0.01 * l2_reg  # L2正则化权重
        loss.backward()
        optimizer.step()

        # 保存loss值用于返回
        loss_value = loss.item()

        del ratings_pred
        return model_client, loss_value  # 修改此处，返回loss值

    def aggregate_clients_params(self, round_user_params):  #在一轮训练中接收客户端模型的参数，对这些参数进行聚合操作，然后将聚合后的结果存储在服务器端。
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():   #遍历 round_user_params 字典的所有键，这些键代表不同的用户。
            # load a user's parameters.
            user_params = round_user_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)

        # train the item representation learning module.
        # item_cv_content = torch.tensor(item_cv_features)
        # item_text_content = torch.tensor(item_text_features)
        # target = self.server_model_param['embedding_item.weight'].data
        # if self.config['use_cuda'] is True:
        #     item_cv_content = item_cv_content.cuda()
        #     item_text_content = item_text_content.cuda()
        #     target = target.cuda()
        # self.server_model.train()
        # for epoch in range(self.config['server_epoch']):
        #     self.server_opt.zero_grad()
        #     logit_rep = self.server_model(item_cv_content, item_text_content)
        #     loss = self.server_crit(logit_rep, target)
        #     loss.backward()
        #     self.server_opt.step()

        # store the global item representation learned by server model.
        # self.server_model.eval()
        # with torch.no_grad():
        #     global_item_rep = self.server_model(item_cv_content, item_text_content)
        # self.server_model_param['global_item_rep'] = global_item_rep



    def fed_train_a_round(self, user_ids, all_train_data, round_id):
        """在单轮联邦训练中添加进度条显示"""
        # 引入tqdm显示进度条
        # 选择参与者
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(user_ids, num_participants, replace=False)
        else:
            participants = np.random.choice(user_ids, self.config['clients_sample_num'], replace=False)

        # 初始化储存所有损失
        all_losses = []

        # 存储当前轮次参与者的模型参数
        round_participant_params = {}

        # 使用tqdm显示进度条
        for user in tqdm(participants, desc=f"轮次 {round_id} 训练中", ncols=100):
            # 为每个用户创建新的模型副本
            model_client = copy.deepcopy(self.client_model)

            # 加载模型参数
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.client_model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item.weight'] = copy.deepcopy(
                    self.server_model_param['embedding_item.weight'].data).cuda()
                model_client.load_state_dict(user_param_dict)

                # 清理中间变量
                del user_param_dict
                torch.cuda.empty_cache()  # 清理CUDA缓存

            # 创建优化器
            optimizer = torch.optim.Adam(model_client.parameters(), lr=self.config['lr_client'],
                                         weight_decay=self.config['l2_regularization'],amsgrad=True)

            # 加载用户训练数据
            user_train_data = all_train_data[user]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()

            # 用户训练损失
            user_losses = []

            # 更新客户端模型
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, batch_loss = self.fed_train_single_batch(model_client, batch, optimizer, round_id)
                    user_losses.append(batch_loss)

            # 添加用户损失到总损失列表
            all_losses.extend(user_losses)

            # 获取客户端模型参数
            client_param = model_client.state_dict()

            # 存储客户端模型的本地参数用于个性化
            self.client_model_params[user] = {}
            for key in client_param.keys():
                # 使用clone()创建新的CPU张量并分离计算图
                self.client_model_params[user][key] = client_param[key].data.clone().cpu()

            # 存储客户端模型的本地参数用于全局更新
            round_participant_params[user] = {}
            round_participant_params[user] = copy.deepcopy(client_param)

            # 清理每个用户训练完后的变量
            del model_client, optimizer, user_dataloader, client_param, user_losses
            torch.cuda.empty_cache()

        # 计算平均损失
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            print(f"轮次 {round_id} 平均损失: {avg_loss:.4f}")

            # 记录日志
            if hasattr(self, 'logger'):
                self.logger.info(f"轮次 {round_id} 平均损失: {avg_loss:.4f}")
            else:
                import os
                os.makedirs("logs", exist_ok=True)
                with open('logs/training_log.txt', 'a') as f:
                    f.write(f"轮次 {round_id} 平均损失: {avg_loss:.4f}\n")

        print("正在聚合客户端参数...")
        # 聚合服务器端的客户端模型
        self.aggregate_clients_params(round_participant_params)

        # print("正在训练服务器内容模型...")
        # # 使用内容特征优化物品嵌入
        # self.train_server_content_model()

        print("正在更新客户端模型...")
        # 更新所有客户端模型参数
        for user in participants:
            for key in self.server_model_param:
                self.client_model_params[user][key] = copy.deepcopy(self.server_model_param[key]).cpu()


        # 清理中间变量
        del round_participant_params, all_losses
        torch.cuda.empty_cache()

        print(f"轮次 {round_id} 完成!")

    def fed_evaluate(self, evaluate_data, item_cv_features, item_text_features, is_test=True):
        """改进的评估函数，正确处理物品ID映射关系并优化内存使用"""
        print(f"\n开始{'测试集' if is_test else '验证集'}评估")

        # 创建物品ID映射
        item_ids = evaluate_data['iid'].unique()
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        # 复制特征数据以避免修改原始数据
        item_cv_content = item_cv_features.clone().detach()
        item_text_content = item_text_features.clone().detach()

        if self.config['use_cuda']:
            item_cv_content = item_cv_content.cuda()
            item_text_content = item_text_content.cuda()

        # 获取用户ID
        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {}

        for user in tqdm(user_ids, desc="评估用户", ncols=100):
            # 获取当前用户交互过的物品
            user_items = evaluate_data[evaluate_data['uid'] == user]['iid'].values
            user_item_indices = [item_id_to_idx[item_id] for item_id in user_items]

            # 配置用户模型
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())

            # 如果用户有个性化参数，加载它
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()

            # 加载服务器全局物品嵌入
            user_param_dict['embedding_item.weight'] = copy.deepcopy(
                self.server_model_param['embedding_item.weight'].data).cuda()

            user_model.load_state_dict(user_param_dict)
            user_model.eval()

            # 使用较小的批次进行预测，防止内存溢出
            batch_size = 128
            all_preds = []

            for i in range(0, len(item_cv_content), batch_size):
                batch_cv = item_cv_content[i:i + batch_size]
                batch_text = item_text_content[i:i + batch_size]

                with torch.no_grad():
                    batch_pred = user_model(batch_cv, batch_text)

                    # 立即转到CPU并分离计算图
                    all_preds.append(batch_pred.cpu().detach())

            # 合并所有批次的预测结果
            cold_pred = torch.cat(all_preds, dim=0).view(-1)

            # 存储用户的预测结果，使用正确的物品ID映射
            user_item_preds[user] = {item_ids[i]: cold_pred[i].item() for i in range(len(item_ids))}

            # 清理变量
            del user_model, user_param_dict, all_preds, cold_pred
            torch.cuda.empty_cache()

        # 计算评估指标
        recall, precision, ndcg = compute_metrics_with_mapping(
            evaluate_data,
            user_item_preds,
            self.config['recall_k'],
            is_test=is_test
        )

        # 清理大型变量
        del user_item_preds, item_cv_content, item_text_content
        torch.cuda.empty_cache()

        return recall, precision, ndcg
