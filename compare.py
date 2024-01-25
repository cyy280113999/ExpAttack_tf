import csv
import numpy as np



def main():
    # 读取CSV文件
    file_path = 'adv_inc3.csv'  # 请替换成你的CSV文件路径
    with open(file_path, 'r') as file:
        # 使用csv.reader读取数据
        reader = csv.reader(file)
        
        # 将数据转换为二维列表
        data = [row for row in reader]

    # 将数据转换为NumPy数组（如果需要的话）
    data = np.array(data, dtype=object)
    head_line = data[0]
    model_names = head_line[1:]
    data = data[1:]
    method_names = data[:,0]
    
    data = data[:,1:]
    # 定义一个替换函数
    def replace_percentage(value):
        # 将百分比符号 "%" 替换为其他值
        return float(value.replace('%', ''))/100

    # 使用 np.vectorize 创建一个ufunc
    ufunc_replace_percentage = np.vectorize(replace_percentage)
    data = ufunc_replace_percentage(data).astype(np.float32)

    # 打印读取到的数据
    print("读取到的数据:")
    print(data)
    
    # 假设 data 是一个二维数组，行表示不同方法的实验结果，列表示不同实验的数据
    # 例如，data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 计算每列的最大值和最小值
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # 归一化每行数据
    normalized_data = (data - min_vals) / (max_vals - min_vals)

    # 计算每行方法的平均实验结果
    average_results = np.mean(normalized_data, axis=1)

    print("每列的最大值:")
    print(max_vals)
    print("\n每列的最小值:")
    print(min_vals)
    print("\n归一化后的数据:")
    print(normalized_data)
    print("\n每行方法的平均实验结果:")
    print(average_results)

if __name__ == '__main__':
    main()