import torch
from engine import Engine


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.num_items_train = config['num_items_train']
        self.num_items_test = config['num_items_test']  # 添加测试集物品数量
        self.num_items_vali = config['num_items_vali']  # 添加验证集物品数量
        self.latent_dim = config['latent_dim']
        self.relu = torch.nn.ReLU()

        # 保留embedding_item，但改变其用途
        # self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Linear(in_features=768 * 2, out_features=self.latent_dim)

        # 创建多层网络结构
        self.fc_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # 解析层配置
        if isinstance(config['client_model_layers'], str):
            layers = [int(x) for x in config['client_model_layers'].split(',')]
        else:
            layers = config['client_model_layers']

        # 构建网络层（第一层处理latent_dim维度的输入）
        input_dim = self.latent_dim
        for output_dim in layers:
            self.fc_layers.append(torch.nn.Linear(input_dim, output_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        # 最终输出层
        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
        self.dropout = torch.nn.Dropout(0.2)
        self.logistic = torch.nn.Sigmoid()
        self.init_weights()

    def forward(self, item_cv, item_txt):
        # 先通过embedding_item处理特征
        item_fet = self.relu(torch.cat([item_cv, item_txt], dim=-1)).cuda()
        item_embedding = self.embedding_item(item_fet)

        # 通过多层网络
        vector = item_embedding
        for idx in range(len(self.fc_layers)):
            vector = self.fc_layers[idx](vector)
            vector = self.batch_norms[idx](vector)
            vector = self.relu(vector)
            vector = self.dropout(vector)

        # 最终输出
        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        if self.training:
            max_idx = self.num_items_train
        else:
            # 根据评估阶段使用不同的最大索引
            max_idx = self.num_items_test  # 或 self.num_items_vali

        rating = torch.clamp(rating, 0, max_idx - 1)
        return rating

    # def cold_predict(self, item_embedding):
    #     # 处理冷启动预测
    #     user_embedding = self.embedding_user(torch.tensor([0] * item_embedding.shape[0]).cuda())
    #     vector = item_embedding  # 直接使用传入的item_embedding
    #
    #     # 通过多层网络
    #     for idx in range(len(self.fc_layers)):
    #         vector = self.fc_layers[idx](vector)
    #         vector = self.batch_norms[idx](vector)
    #         vector = self.relu(vector)
    #         vector = self.dropout(vector)
    #
    #     logits = self.affine_output(vector)
    #     rating = self.logistic(logits)
    #     rating = torch.clamp(rating, 0, self.num_items_test - 1)  # 用于测试集
    #     return rating

    def init_weights(self):
        """
        Initialize weights for all neural network layers using Xavier initialization
        For Linear layers: weights using xavier_normal_, biases to 0
        For Embedding layers: weights using normal distribution
        For BatchNorm layers: weights to 1, biases to 0
        """
        # Initialize embedding_item (Linear layer)
        torch.nn.init.xavier_normal_(self.embedding_item.weight)
        torch.nn.init.zeros_(self.embedding_item.bias)

        # Initialize FC layers
        for layer in self.fc_layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        # Initialize BatchNorm layers
        for bn in self.batch_norms:
            torch.nn.init.ones_(bn.weight)
            torch.nn.init.zeros_(bn.bias)

        # Initialize final output layer
        torch.nn.init.xavier_normal_(self.affine_output.weight)
        torch.nn.init.zeros_(self.affine_output.bias)


    def load_pretrain_weights(self):
        pass


# class Server(torch.nn.Module):
#     def __init__(self, config):
#         super(Server, self).__init__()
#         self.config = config
#         self.content_dim = config['content_dim']
#         self.latent_dim = config['latent_dim']
#
#         # 创建多层网络结构
#         self.fc_layers = torch.nn.ModuleList()
#         self.batch_norms = torch.nn.ModuleList()
#
#         # 解析层配置
#         if isinstance(config['server_model_layers'], str):
#             layers = [int(x) for x in config['server_model_layers'].split(',')]
#         else:
#             layers = config['server_model_layers']
#
#         # 第一层处理拼接后的特征 (768*2 = 1536)
#         input_dim = self.content_dim * 2
#
#         # 构建网络层
#         for output_dim in layers:
#             self.fc_layers.append(torch.nn.Linear(input_dim, output_dim))
#             self.batch_norms.append(torch.nn.BatchNorm1d(output_dim))
#             input_dim = output_dim
#
#         # 最终输出层
#         self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=self.latent_dim)
#         self.dropout = torch.nn.Dropout(0.2)
#         self.logistic = torch.nn.Tanh()
#
#     def forward(self, item_cv_feat, item_txt_feat):
#         vector = torch.cat([item_cv_feat, item_txt_feat], dim=-1).cuda()
#
#         for idx in range(len(self.fc_layers)):
#             vector = self.fc_layers[idx](vector)
#             vector = self.batch_norms[idx](vector)
#             vector = torch.nn.ReLU()(vector)
#             vector = self.dropout(vector)
#
#         logits = self.affine_output(vector)
#         rating = self.logistic(logits)
#         return rating
#
#     def init_weight(self):
#         pass
#
#     def load_pretrain_weights(self):
#         pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.client_model = Client(config)
        # self.server_model = Server(config)
        if config['use_cuda'] is True:
            # use_cuda(True, config['device_id'])
            self.client_model.cuda()
            # self.server_model.cuda()
        super(MLPEngine, self).__init__(config)
