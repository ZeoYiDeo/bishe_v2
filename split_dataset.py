import pandas as pd
import numpy as np
import copy


def split_data_for_federated_recommendation(file_path, test_ratio=0.2):
    """
    为联邦推荐系统划分数据集

    Args:
        file_path: CSV文件路径
        test_ratio: 测试集比例
        valid_ratio: 验证集比例

    Returns:
        训练集、验证集和测试集的字典
    """
    # 加载数据
    df = pd.read_csv(file_path,header=None,names=['iid', 'uid'], usecols=[0, 1])
    print(f"原始数据大小: {len(df)}，用户数: {df['uid'].nunique()}，物品数: {df['iid'].nunique()}")

    # 按用户分组
    user_groups = df.groupby('uid')

    train_data, test_data = [], []

    # 为每个用户划分交互数据
    for user_id, user_data in user_groups:
        # 如果有时间戳，按时间排序
        if 'timestamp' in user_data.columns:
            user_data = user_data.sort_values('timestamp')
        else:
            # 没有时间戳就随机打乱
            user_data = user_data.sample(frac=1, random_state=42)

        n_interactions = len(user_data)

        # 对于交互很少的用户，全部放入训练集
        if n_interactions < 5:
            train_data.append(user_data)
            continue

        # 计算划分大小
        n_test = max(int(test_ratio * n_interactions), 1)
        n_train = n_interactions - n_test

        # 确保每个集合至少有一条数据
        if n_train < 1:
            n_train = 1
            n_test = min(1, n_interactions - n_train)

        # 划分用户数据 - 按时间顺序，早期数据用于训练
        user_train = user_data.iloc[:n_train]
        user_test = user_data.iloc[n_train :]

        train_data.append(user_train)
        test_data.append(user_test)

    # 合并所有用户的数据
    train_df = pd.concat(train_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)

    print(f"训练集: {len(train_df)} 条交互 ({len(train_df) / len(df):.2%})")
    print(f"测试集: {len(test_df)} 条交互 ({len(test_df) / len(df):.2%})")

    return {
        'train': train_df,
        'test': test_df
    }

if __name__ == '__main__':
    # 加载并划分数据
    data_splits = split_data_for_federated_recommendation('../data/DY/DY_pair_mini.csv')

    # 保存划分后的数据
    data_splits['train'].to_csv('../data/DY/train_mini.csv', index=False)
    data_splits['test'].to_csv('../data/DY/test_mini.csv', index=False)

    # 检查用户和物品的分布
    for split_name in ['train', 'test']:
        split = data_splits[split_name]
        print(f"{split_name} - 用户数: {split['uid'].nunique()}, 物品数: {split['iid'].nunique()}")
