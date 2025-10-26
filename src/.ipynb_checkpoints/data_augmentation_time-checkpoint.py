# -*- coding: utf-8 -*-

import copy
import random
import itertools
import numpy as np

def get_var(tlist):
    length = len(tlist)
    total = 0
    diffs = []
    if length <= 1:
        return 0
    for i in range(length - 1):
        diff = abs(tlist[i + 1] - tlist[i])
        diffs.append(diff)
        total = total + diff
    if len(diffs) == 0:
        return 0
    avg_diff = total / len(diffs)
    total = sum((diff - avg_diff) ** 2 for diff in diffs)
    result = total / len(diffs)
    return result

class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, args, similarity_model):
        self.short_seq_data_aug_methods = None
        self.augment_threshold = args.augment_threshold
        self.augment_type_for_short = args.augment_type_for_short
        if self.augment_threshold == -1:
            self.data_augmentation_methods = [Crop(args.crop_mode, args.crop_rate),
                                              Mask(args.mask_mode, args.mask_rate),
                                              Reorder(args.reorder_mode, args.reorder_rate),
                                              Pooling(args.pooling_mode, args.pooling_rate),
                                              Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                     args.max_insert_num_per_pos),
                                              Substitute(similarity_model, args.substitute_mode, args.substitute_rate)]
            print("Total augmentation numbers: ", len(self.data_augmentation_methods))
        elif self.augment_threshold > 0:
            print("short sequence augment type:", self.augment_type_for_short)
            self.short_seq_data_aug_methods = []
            if 'S' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(
                    Substitute(similarity_model, args.substitute_mode, args.substitute_rate))
            if 'I' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(
                    Insert(similarity_model, args.insert_mode, args.insert_rate, args.max_insert_num_per_pos))
            if 'M' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Mask(args.mask_mode, args.mask_rate))
            if 'R' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Reorder(args.reorder_mode, args.reorder_rate))
            if 'C' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Crop(args.crop_mode, args.crop_rate))
            if 'P' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Pooling(args.pooling_mode, args.pooling_rate))
            if len(self.augment_type_for_short) == 6:
                print("all aug set for short sequences")
            self.long_seq_data_aug_methods = [Crop(args.crop_mode, args.crop_rate),
                                              Mask(args.mask_mode, args.mask_rate),
                                              Reorder(args.reorder_mode, args.reorder_rate),
                                              Pooling(args.pooling_mode, args.pooling_rate),
                                              Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                     args.max_insert_num_per_pos),
                                              Substitute(similarity_model, args.substitute_mode, args.substitute_rate)]
            print("Augmentation methods for Long sequences:", len(self.long_seq_data_aug_methods))
            print("Augmentation methods for short sequences:", len(self.short_seq_data_aug_methods))
        else:
            raise ValueError("Invalid data type.")

    def __call__(self, item_sequence, time_sequence):
        if self.augment_threshold == -1:
            # randint generate int x in range: a <= x <= b
            augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
            augment_method = self.data_augmentation_methods[augment_method_idx]
            return augment_method(item_sequence, time_sequence)
        elif self.augment_threshold > 0:
            seq_len = len(item_sequence)
            if seq_len > self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.long_seq_data_aug_methods) - 1)
                augment_method = self.long_seq_data_aug_methods[augment_method_idx]
                return augment_method(item_sequence, time_sequence)
            elif seq_len <= self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.short_seq_data_aug_methods) - 1)
                augment_method = self.short_seq_data_aug_methods[augment_method_idx]
                return augment_method(item_sequence, time_sequence)


def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]


class Insert(object):
    """
    Insert similar items every time call.
    Priority is given to places with large time intervals.
    maximum: Insert at larger time intervals
    minimum: Insert at smaller time intervals
    """

    def __init__(self, item_similarity_model, mode, insert_rate=0.4, max_insert_num_per_pos=1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.mode = mode
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        insert_nums = max(int(self.insert_rate * len(copied_sequence)), 1)

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        insert_idx = []
        for i in range(insert_nums):
            temp = diff_sorted[i]
            insert_idx.append(temp)

        """
        The index of time_diff is 1 smaller than the item. 
        The item should be inserted to the right of item_index. 
        Put the original item first in each cycle, so that the inserted item is inserted to the right of the original item
        """
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):

            inserted_sequence += [item]

            if index in insert_idx:
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos / insert_nums)))
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(item, top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(item, top_k=top_k, with_score=True)
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    inserted_sequence += self.item_similarity_model.most_similar(item, top_k=top_k)

        return inserted_sequence


class Substitute(object):
    """
    Substitute with similar items
    maximum: Substitute items with larger time interval
    minimum: Substitute items with smaller time interval
    """

    def __init__(self, item_similarity_model, mode, substitute_rate=0.1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        if len(copied_sequence) <= 1:
            return copied_sequence
        substitute_nums = max(int(self.substitute_rate * len(copied_sequence)), 1)

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)

        diff_sorted = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        substitute_idx = []
        for i in range(substitute_nums):
            temp = diff_sorted[i]
            substitute_idx.append(temp)

        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index], with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index], with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:
                copied_sequence[index] = copied_sequence[index] = \
                    self.item_similarity_model.most_similar(copied_sequence[index])[0]
        return copied_sequence


class Crop(object):
    """
    maximum: Crop subsequences with the maximum time interval variance
    minimum: Crop subsequences with the minimum time interval variance
    """

    def __init__(self, mode, tao=0.2):
        self.tao = tao
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length <= 2:
            return [copied_sequence[start_index]]

        cropped_vars = []
        crop_index = []
        for i in range(len(item_sequence)):
            if len(item_sequence) - i - sub_seq_length >= 0:
                left_index = len(item_sequence) - i - sub_seq_length
                right_index = left_index + sub_seq_length
                temp_time_sequence = time_sequence[left_index:right_index - 1]
                temp_var = get_var(temp_time_sequence)

                cropped_vars.append(temp_var)
                crop_index.append(left_index)
        temp = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            temp = cropped_vars.index(max(cropped_vars))
        if self.mode == 'minimum':
            temp = cropped_vars.index(min(cropped_vars))
        start_index = crop_index.index(temp)

        cropped_sequence = copied_sequence[start_index:start_index + sub_seq_length]
        return cropped_sequence


class Mask(object):
    """
    Randomly mask k items given a sequence
    maximum: Mask items with larger time interval
    minimum: Mask items with smaller time interval
    """

    def __init__(self, mode, gamma=0.7):
        self.gamma = gamma
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]

        if len(copied_sequence) <= 1:
            return copied_sequence

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)

        diff_sorted = []
        assert self.mode in ['maximum', 'minimum', 'random']
        if self.mode == 'random':
            copied_sequence = copy.deepcopy(item_sequence)
            mask_nums = int(self.gamma * len(copied_sequence))
            mask = [0 for i in range(mask_nums)]
            mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
            for idx, mask_value in zip(mask_idx, mask):
                copied_sequence[idx] = mask_value
            return copied_sequence
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        mask_idx = []
        for i in range(mask_nums):
            temp = diff_sorted[i]
            mask_idx.append(temp)

        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """
    Randomly shuffle a continuous sub-sequence
    maximum: Reorder subsequences with the maximum time interval variance
    minimum: Reorder subsequences with the minimum variance of time interval
    """

    def __init__(self, mode, beta=0.2):
        self.beta = beta
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        if sub_seq_length < 2:
            return copied_sequence

        cropped_vars = []
        crop_index = []
        for i in range(len(item_sequence)):
            if len(item_sequence) - i - sub_seq_length >= 0:
                left_index = len(item_sequence) - i - sub_seq_length
                right_index = left_index + sub_seq_length
                temp_time_sequence = time_sequence[left_index:right_index - 1]
                temp_var = get_var(temp_time_sequence)

                cropped_vars.append(temp_var)
                crop_index.append(left_index)
        temp = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            temp = cropped_vars.index(max(cropped_vars))
        if self.mode == 'minimum':
            temp = cropped_vars.index(min(cropped_vars))
        start_index = crop_index.index(temp)

        sub_seq = copied_sequence[start_index:start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq

class Pooling(object):

    def __init__(self, mode, omega=0.5):
        self.mode = mode
        self.omega = omega

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)                   # 深度拷贝复制序列
        sub_seq_length = int(self.omega * len(copied_sequence))          # 根据池化率omega获取子序列长度
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)   # 先随机获取子序列的开始位置
        if len(copied_sequence) <=2:                                     # 序列不够长返回序列开始的位置
            return [copied_sequence[start_index]]
        # 根据时间项目序列来确定子序列的开始位置，即结合时间间隔因素
        cropped_vars = []
        crop_index = []
        for i in range(len(item_sequence)):
            if len(item_sequence) - i - sub_seq_length >= 0:
                left_index = len(item_sequence) - i - sub_seq_length
                right_index = left_index + sub_seq_length
                temp_time_sequence = time_sequence[left_index:right_index - 1]
                temp_var = get_var(temp_time_sequence)

                cropped_vars.append(temp_var)
                crop_index.append(left_index)
        temp = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            temp = cropped_vars.index(max(cropped_vars))
        if self.mode == 'minimum':
            temp = cropped_vars.index(min(cropped_vars))
        start_index = crop_index.index(temp)                              # 最终由时间间隔确定子序列的开始位置，并且子序列具有最小标准差

        # 计算子序列的平均值，用于生成虚拟项目p
        sub_seq = copy.deepcopy(copied_sequence[start_index:start_index + sub_seq_length])       # 获取子序列
        pooled_item = np.mean(sub_seq, axis=0)                            # 进行平均池化
        # 将生成的项目替换掉原子序列，获取增强后的序列
        new_seq = copied_sequence[:start_index] + [pooled_item] + copied_sequence[start_index + sub_seq_length:]
        return new_seq