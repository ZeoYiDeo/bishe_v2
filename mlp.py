import torch
from engine import Engine


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
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
        self.dropout = torch.nn.Dropout(0.6)
        self.logistic = torch.nn.Sigmoid()
        self.init_weights()

    def forward(self, item_cv, item_txt):
        # 先通过embedding_item处理特征
        item_fet = self.relu(torch.cat([item_cv, item_txt], dim=-1)).cuda()
        item_embedding = self.embedding_item(item_fet)

        # 通过多层网络
        vector = item_embedding
        for idx in range(len(self.fc_layers)):
            vector = self.dropout(self.fc_layers[idx](vector))
            vector = self.batch_norms[idx](vector)
            vector = self.relu(vector)
            vector = self.dropout(vector)

        # 最终输出
        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        return rating


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








class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.client_model = Client(config)

        if config['use_cuda'] is True:

            self.client_model.cuda()

        super(MLPEngine, self).__init__(config)
