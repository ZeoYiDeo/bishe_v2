import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data_with_complete_coverage(file_path, max_train_ratio=0.8):
    """
    将数据集划分为训练集、测试集和验证集
    确保:
    1. 训练集、测试集和验证集中都包含所有物品ID
    2. 测试集和验证集中每个物品ID和用户ID至少出现一次
    3. 训练集比例不超过指定值(默认80%)

    参数:
    file_path: CSV文件路径
    max_train_ratio: 训练集最大比例，默认为0.8

    返回:
    train_data, test_data, val_data: 划分后的数据集
    """
    # 读取CSV文件
    data = pd.read_csv(file_path,header=None, usecols=[0, 1])

    # 假设第一列是物品id，第二列是用户id
    item_column = data.columns[0]
    user_column = data.columns[1]

    # 获取所有唯一的物品ID和用户ID
    unique_items = data[item_column].unique()
    unique_users = data[user_column].unique()

    print(f"数据集中共有 {len(unique_items)} 个唯一物品ID和 {len(unique_users)} 个唯一用户ID")

    # 为测试集、验证集和训练集保留样本
    test_samples = []
    val_samples = []
    train_samples = []
    remaining_data = data.copy()

    # 用于记录已经在各数据集中的物品ID和用户ID
    test_items_covered = set()
    test_users_covered = set()
    val_items_covered = set()
    val_users_covered = set()
    train_items_covered = set()

    # 第一阶段：确保每个物品ID在测试集、验证集和训练集中都有至少一条记录
    for item_id in unique_items:
        item_data = data[data[item_column] == item_id]

        if len(item_data) >= 3:
            # 如果该物品至少有3条记录，则各集合各取1条
            test_sample = item_data.sample(1)
            test_items_covered.add(item_id)
            test_users_covered.add(test_sample[user_column].iloc[0])
            test_samples.append(test_sample)

            # 从剩余记录中挑选验证集样本
            remaining_item_data = item_data.drop(test_sample.index)
            val_sample = remaining_item_data.sample(1)
            val_items_covered.add(item_id)
            val_users_covered.add(val_sample[user_column].iloc[0])
            val_samples.append(val_sample)

            # 从剩余记录中挑选训练集样本
            remaining_item_data = remaining_item_data.drop(val_sample.index)
            train_sample = remaining_item_data.sample(1)
            train_items_covered.add(item_id)
            train_samples.append(train_sample)

            # 从原始数据中删除这些已选择的样本
            remaining_data = remaining_data.drop(
                index=test_sample.index.union(val_sample.index).union(train_sample.index))

        elif len(item_data) == 2:
            # 如果该物品只有2条记录，为测试集和验证集各取1条，训练集复制一条
            test_sample = item_data.sample(1)
            test_items_covered.add(item_id)
            test_users_covered.add(test_sample[user_column].iloc[0])
            test_samples.append(test_sample)

            val_sample = item_data.drop(test_sample.index)
            val_items_covered.add(item_id)
            val_users_covered.add(val_sample[user_column].iloc[0])
            val_samples.append(val_sample)

            # 为训练集复制一条记录（从测试集或验证集）
            train_samples.append(test_sample.copy())  # 复制测试集的样本
            train_items_covered.add(item_id)

            # 从原始数据中删除这些已选择的样本
            remaining_data = remaining_data.drop(index=item_data.index)

        elif len(item_data) == 1:
            # 如果该物品只有1条记录，复制到所有三个集合
            test_samples.append(item_data)
            val_samples.append(item_data.copy())
            train_samples.append(item_data.copy())

            test_items_covered.add(item_id)
            val_items_covered.add(item_id)
            train_items_covered.add(item_id)

            user_id = item_data[user_column].iloc[0]
            test_users_covered.add(user_id)
            val_users_covered.add(user_id)

            # 从原始数据中删除这条样本
            remaining_data = remaining_data.drop(item_data.index)

    # 第二阶段：确保每个用户ID在测试集和验证集中都有至少一条记录
    for user_id in unique_users:
        if user_id in test_users_covered and user_id in val_users_covered:
            continue  # 该用户已经在两个集合中都有样本

        user_data = remaining_data[remaining_data[user_column] == user_id]

        if len(user_data) == 0:
            continue  # 没有可用的样本了

        if user_id not in test_users_covered and len(user_data) > 0:
            # 为测试集添加该用户的一个样本
            test_sample = user_data.sample(1)
            test_users_covered.add(user_id)
            test_items_covered.add(test_sample[item_column].iloc[0])
            test_samples.append(test_sample)
            user_data = user_data.drop(test_sample.index)
            remaining_data = remaining_data.drop(test_sample.index)

        if user_id not in val_users_covered and len(user_data) > 0:
            # 为验证集添加该用户的一个样本
            val_sample = user_data.sample(1)
            val_users_covered.add(user_id)
            val_items_covered.add(val_sample[item_column].iloc[0])
            val_samples.append(val_sample)
            remaining_data = remaining_data.drop(val_sample.index)

    # 合并测试集、验证集和初步的训练集样本
    test_data = pd.concat(test_samples) if test_samples else pd.DataFrame(columns=data.columns)
    val_data = pd.concat(val_samples) if val_samples else pd.DataFrame(columns=data.columns)
    initial_train_data = pd.concat(train_samples) if train_samples else pd.DataFrame(columns=data.columns)

    # 将剩余数据添加到训练集
    train_data = pd.concat([initial_train_data, remaining_data])

    # 检查训练集是否包含所有物品ID，如果不包含则从测试集和验证集复制样本
    missing_items_in_train = set(unique_items) - set(train_data[item_column].unique())

    if missing_items_in_train:
        print(f"训练集缺少 {len(missing_items_in_train)} 个物品ID，将从测试集或验证集复制样本")

        for item_id in missing_items_in_train:
            # 优先从测试集获取样本
            test_item_sample = test_data[test_data[item_column] == item_id].sample(1) if not test_data[
                test_data[item_column] == item_id].empty else None
            val_item_sample = val_data[val_data[item_column] == item_id].sample(1) if not val_data[
                val_data[item_column] == item_id].empty else None

            sample_to_copy = test_item_sample if test_item_sample is not None else val_item_sample

            if sample_to_copy is not None:
                train_data = pd.concat([train_data, sample_to_copy.copy()])

    # 检查训练集比例是否超过最大值
    total_size = len(data)
    train_ratio = len(train_data) / total_size

    if train_ratio > max_train_ratio:
        # 如果训练集比例过大，需要将一部分训练数据移至测试集和验证集
        extra_samples_needed = int(total_size * (train_ratio - max_train_ratio))

        # 从初始训练数据中排除用于保证物品ID覆盖的样本，只从多余部分抽样
        core_train_item_samples = initial_train_data[item_column].unique()

        # 找出那些在训练集中出现次数较多的样本
        item_counts = train_data[item_column].value_counts()
        user_counts = train_data[user_column].value_counts()

        # 优先从高频物品/用户中抽样
        potential_extra_samples = train_data[
            (train_data[item_column].map(item_counts) > 1) &
            (train_data[user_column].map(user_counts) > 1)
            ]

        if len(potential_extra_samples) >= extra_samples_needed:
            extra_samples = potential_extra_samples.sample(extra_samples_needed)
            # 将额外样本平均分配到测试集和验证集
            half_point = len(extra_samples) // 2
            extra_test = extra_samples.iloc[:half_point]
            extra_val = extra_samples.iloc[half_point:]

            test_data = pd.concat([test_data, extra_test])
            val_data = pd.concat([val_data, extra_val])

            # 从训练集中移除这些样本
            train_data = train_data.drop(extra_samples.index)

    # 打印每个集合的大小和比例
    print(f"训练集大小: {len(train_data)} ({len(train_data) / total_size:.2%})")
    print(f"测试集大小: {len(test_data)} ({len(test_data) / total_size:.2%})")
    print(f"验证集大小: {len(val_data)} ({len(val_data) / total_size:.2%})")

    # 验证物品ID和用户ID覆盖情况
    train_items = set(train_data[item_column].unique())
    test_items = set(test_data[item_column].unique())
    val_items = set(val_data[item_column].unique())

    train_users = set(train_data[user_column].unique())
    test_users = set(test_data[user_column].unique())
    val_users = set(val_data[user_column].unique())

    print("\n物品ID覆盖情况:")
    print(f"总物品ID数: {len(unique_items)}")
    print(f"训练集中物品ID数: {len(train_items)} ({len(train_items) / len(unique_items):.2%})")
    print(f"测试集中物品ID数: {len(test_items)} ({len(test_items) / len(unique_items):.2%})")
    print(f"验证集中物品ID数: {len(val_items)} ({len(val_items) / len(unique_items):.2%})")

    print("\n用户ID覆盖情况:")
    print(f"总用户ID数: {len(unique_users)}")
    print(f"训练集中用户ID数: {len(train_users)} ({len(train_users) / len(unique_users):.2%})")
    print(f"测试集中用户ID数: {len(test_users)} ({len(test_users) / len(unique_users):.2%})")
    print(f"验证集中用户ID数: {len(val_users)} ({len(val_users) / len(unique_users):.2%})")

    # 最终验证是否所有物品ID都在三个集合中
    assert len(train_items) == len(unique_items), "训练集应包含所有物品ID"
    assert len(test_items) == len(unique_items), "测试集应包含所有物品ID"
    assert len(val_items) == len(unique_items), "验证集应包含所有物品ID"

    print("\n✓ 所有数据集均包含全部物品ID")

    # 检查是否所有用户ID都在测试集和验证集中至少出现一次
    missing_users_test = set(unique_users) - test_users
    missing_users_val = set(unique_users) - val_users

    if missing_users_test or missing_users_val:
        print("\n警告: 存在未覆盖的用户ID")
        if missing_users_test:
            print(f"测试集中缺少 {len(missing_users_test)} 个用户ID")
        if missing_users_val:
            print(f"验证集中缺少 {len(missing_users_val)} 个用户ID")
    else:
        print("✓ 所有用户ID均在测试集和验证集中至少出现一次")

    return train_data, test_data, val_data


# 使用示例
if __name__ == "__main__":
    file_path = "../data/Bili_Food/Bili_Food_pair.csv"
    train_data, test_data, val_data = split_data_with_complete_coverage(file_path)

    # 保存划分后的数据集（可选）
    train_data.to_csv("../data/Bili_Food/train_data.csv", index=False)
    test_data.to_csv("../data/Bili_Food/test_data.csv", index=False)
    val_data.to_csv("../data/Bili_Food/val_data.csv", index=False)
