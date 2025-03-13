import torch
from utils import *
import numpy as np
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

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

    def lambda_loss(self, prediction, label, lambda_val=0.1):
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


    def fed_train_single_batch(self, model_client, batch_data, optimizer,scheduler):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings,cv,ct = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4]
        ratings = ratings.float()
        # reg_item_embedding = copy.deepcopy(self.server_model_param['global_item_rep'])  #服务器的全局物品表示，用于计算后续的2正则化项

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            # reg_item_embedding = reg_item_embedding.cuda()
 #分别为客户端的优化器、用户嵌入的优化器、物品嵌入的优化器
        # update score function.
        optimizer.zero_grad()
        ratings_pred = model_client(cv,ct)
        # loss = self.client_crit(ratings_pred.view(-1), ratings)  #计算损失值。self.client_crit 是一个损失函数（从之前的代码可知，可能是 torch.nn.BCELoss()），ratings_pred.view(-1) 将预测的评分展平为一维张量，然后与真实的评分 ratings 计算损失。
        loss = self.lambda_loss(ratings_pred.view(-1), ratings)
        print("\r loss is {:.4f}".format(loss.item()), end='')
        # regularization_term = compute_regularization(model_client, reg_item_embedding)  #计算2正则化项。
        # loss += self.config['reg'] * regularization_term
        loss.backward()
        optimizer.step()
        scheduler.step()
        del loss, ratings_pred
        return model_client

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
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(user_ids, num_participants, replace=False)
        else:
            participants = np.random.choice(user_ids, self.config['clients_sample_num'], replace=False)

        # initialize server parameters for the first round.
        # if round_id == 0:
        #     item_cv_content = torch.tensor(item_cv_features)
        #     item_text_content = torch.tensor(item_text_features)
        #     if self.config['use_cuda'] is True:
        #         item_cv_content = item_cv_content.cuda()
        #         item_text_content = item_text_content.cuda()
        #     self.server_model.eval()
        #     with torch.no_grad():
        #         global_item_rep = self.server_model(item_cv_content, item_text_content)
        #     self.server_model_param['global_item_rep'] = global_item_rep

        # store users' model parameters of current round.
        round_participant_params = {}
        # perform model update for each participated user.
        for user in participants:
            # copy the client model architecture from self.client_model
            model_client = copy.deepcopy(self.client_model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated item embedding from server.
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.client_model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating score function.
            optimizer = torch.optim.Adam(model_client.parameters(),lr=self.config['lr_client'],weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_u is responsible for updating user embedding. # Item optimizer
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['total_steps'])

            # load current user's training data and instance a train loader.
            user_train_data = all_train_data[user]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)  #检查批次数据的类型是否为 torch.LongTensor，如果不是则会抛出 AssertionError。
                    model_client = self.fed_train_single_batch(model_client, batch, optimizer,scheduler)
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = {}
            for key in client_param.keys():
                if key != 'embedding_item.weight':
                    self.client_model_params[user][key] = client_param[key].data.cpu()
            # store client models' local parameters for global update.
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item.weight'] = client_param['embedding_item.weight'].data.cpu()
        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params)
        del round_participant_params

    def fed_evaluate(self, evaluate_data, item_cv_features, item_text_features, is_test=True):
        """修改后的评估函数，不使用item_ids_map"""
        print(f"\n开始{'测试集' if is_test else '验证集'}评估")
        print(f"评估数据大小: {len(evaluate_data)}")
        print(f"物品特征数量: {len(item_cv_features)}")

        item_cv_content = torch.tensor(item_cv_features)
        item_text_content = torch.tensor(item_text_features)

        # 获取用户ID
        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {}

        for user in user_ids:
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                cold_pred = user_model(item_cv_content, item_text_content)
                user_item_preds[user] = cold_pred.view(-1)

        # 使用修改后的评估指标计算函数
        recall, precision, ndcg = compute_metrics_without_mapping(
            evaluate_data,
            user_item_preds,
            self.config['recall_k'],
            is_test=is_test
        )
        return recall, precision, ndcg