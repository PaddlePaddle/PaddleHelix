import sys
from functools import reduce

import pickle
import numpy as np

import paddle

def load_pkl(file_path):
    """加载 .pkl 文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compare_arrays(array1, array2, path=""):
    """比较两个 numpy 数组是否相等"""
    # print(array1)
    # print(array2)
    if array1.shape != array2.shape:
        return [f"{path}: 形状不同 | array1.shape={array1.shape}, array2.shape={array2.shape}"]
    if not np.array_equal(array1, array2):
        return [f"{path}: 值不同 | array1 和 array2 的值不完全相同"]
    return []

def compare_dicts(dict1, dict2, path=""):
    """递归比较两个字典的键和值"""
    differences = []

    # 检查键是否一致
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    if keys1 != keys2:
        missing_in_dict2 = keys1 - keys2
        missing_in_dict1 = keys2 - keys1
        if missing_in_dict2:
            differences.append(f"{path}: 键 {missing_in_dict2} 在 dict2 中缺失")
        if missing_in_dict1:
            differences.append(f"{path}: 键 {missing_in_dict1} 在 dict1 中缺失")

    # 比较共同键的值
    common_keys = keys1 & keys2
    for key in common_keys:
        new_path = f"{path}.{key}" if path else key
        value1 = dict1[key]
        value2 = dict2[key]

        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            # 如果值是 numpy 数组，比较数组
            differences.extend(compare_arrays(value1, value2, new_path))
        elif isinstance(value1, dict) and isinstance(value2, dict):
            # 如果值是字典，递归比较
            differences.extend(compare_dicts(value1, value2, new_path))
        else:
            # 其他类型直接比较
            if value1 != value2:
                differences.append(f"{new_path}: 值不同 | dict1={value1}, dict2={value2}")

    return differences

def compare_feature_pkl(file1, file2):
    """主函数：加载并比较两个 pkl 文件"""
    dict1 = load_pkl(file1)
    dict2 = load_pkl(file2)

    # 比较两个字典
    differences = compare_dicts(dict1, dict2)

    # 输出差异
    if differences:
        print("发现以下差异：")
        for diff in differences:
            print(diff)
        return False
    else:
        print("两个 pkl 文件内容完全相同。")
        return True

def compare_paddle_files(file1, file2):
    """对比两个 paddle.save 保存的文件"""
    # 加载文件
    data1 = paddle.load(file1)
    data2 = paddle.load(file2)

    # 检查是否为字典
    if isinstance(data1, dict) and isinstance(data2, dict):
        print("Warning: 文件内容是字典，不支持检测，跳过。")
    elif isinstance(data1, paddle.Tensor) and isinstance(data2, paddle.Tensor):
        od1 = data1
        od2 = data2
        if data1.dtype == paddle.bfloat16:
            # 如果是 bfloat16，必须强转 float32 对比
            data1 = data1.astype('float32')
            data2 = data2.astype('float32')

        if paddle.equal_all(data1, data2):
            pass
        else:
            print("原 dtype:", od1.dtype)
            print('转为 float32 不相等：', data1, data2)
            mask = data1 != data2
            product = reduce(lambda x, y: x * y, data1.shape)
            print('不相等的比例：', mask.sum()/product)
            print('原数据：', od1, od2)
            return False
    return True


if __name__ == "__main__":
    # 替换为你的 pkl 文件路径
    # file1 = "file1.pkl"
    # file2 = "file2.pkl"
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # compare_feature_pkl(file1, file2)

    compare_paddle_files(file1, file2)
