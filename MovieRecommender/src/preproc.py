# 导入pandas库以进行数据处理
import pandas as pd
# 导入json库以处理json格式数据
import json
# 导入re库以使用正则表达式
import re

# 定义Netflix函数，处理Netflix数据集
def Netflix(MAX_USER = 500):
    # 创建一个字典d_movie来存储电影ID和标题
    d_movie = dict()
    # 创建一个集合s_movie来存储电影标题，用于去重
    s_movie = set()
    
    # 打开一个文件out_movies.csv，准备写入电影标题
    out_movies = open("../out_movies.csv","w")
    # 写入标题行
    out_movies.write("title\n")

    # 遍历movie_titles.csv文件中的每一行
    for line in open("../movie_titles.csv","r",encoding = 'ISO-8859-1'):
        # 移除行两端的空白并将其拆分为一个列表
        line = line.strip().split(',') 
        # 获取电影ID，将字符串转换为整数
        movie_id = int(line[0])
        # 获取电影标题，并移除双引号
        title = line[2].replace("\"","")
        # 为标题添加双引号，确保格式统一
        title = "\"" + title + "\""
        
        # 将电影ID和标题添加到字典d_movie中
        d_movie[movie_id] = title
        
        # 如果标题已存在于集合s_movie中，跳过本次循环的后续部分
        if title in s_movie:
           continue
        # 将标题添加到集合s_movie中
        s_movie.add(title)
        
        # 将标题写入out_movies.csv文件
        out_movies.write(f"{title}\n")
        
    # 关闭out_movies.csv文件
    out_movies.close()
    
    # 打开一个文件out_grade.csv，准备写入用户评分数据
    out_grade = open("../out_grade.csv","w")
    # 写入标题行
    out_grade.write("user_id,title,grade\n")

    # 定义要处理的文件列表
    files = ["../combined_data_1.txt"]
    # 遍历文件列表中的每个文件
    for f in files:
        # 初始化电影ID为-1
        movie_id = -1
        # 遍历文件中的每一行
        for line in open(f,"r"):
            # 查找冒号的位置
            pos = line.find(":")
            # 如果找到了冒号，意味着有新的用户信息
            if pos!=-1: 
                # 获取用户ID
                movie_id = int(line[:pos])
                # 跳过本次循环的后续部分，直接进入下一轮循环，开始处理新用户的信息
                continue
            # 移除行两端的空白并拆分为一个列表
            line = line.strip().split(',')
            # 获取用户ID，并将字符串转换为整数
            user_id = int(line[0])
            # 获取用户的评分，将字符串转换为整数
            rating = int(line[1])
            
            # 如果用户ID大于设定的最大用户ID，跳过当前行的处理
            if user_id > MAX_USER:
                continue
            # 将用户ID、电影标题和评分写入out_grade.csv文件
            out_grade.write(f"{user_id},{d_movie[movie_id]},{rating}\n")

    # 关闭out_grade.csv文件
    out_grade.close()

# 定义TMDB函数，用于处理TMDB数据集
def TMDB():
    # 构造一个正则表达式模式，用于匹配标题
    pattern = re.compile("[A-Za-z0-9]+")
    # 打开out_genre.csv文件，准备写入电影标题和类型信息
    out_genre = open("../out_genre.csv","w",encoding='utf-8')
    # 写入标题行
    out_genre.write("title,genre\n")
    # 打开out_keyword.csv文件，准备写入电影标题和关键词信息
    out_keyword = open("../out_keyword.csv","w",encoding='utf-8')
    # 写入标题行
    out_keyword.write("title,keyword\n")
    # 打开out_productor.csv文件，准备写入电影标题和制作人信息
    out_productor = open("../out_productor.csv","w",encoding='utf-8')
    # 写入标题行
    out_productor.write("title,productor\n")
    
    # 读取tmdb_5000_movies.csv文件，设置分隔符为逗号
    df = pd.read_csv("../tmdb_5000_movies.csv", sep=",")
    # 列出需要解析的json格式列
    json_columns = ['genres', 'keywords', 'production_companies']
    # 遍历这些列，将json字符串解析为python对象
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    # 筛选出需要的列
    df = df[["genres", "keywords", "original_title","production_companies"]]
    # 遍历数据框中的每一行
    for _, row in df.iterrows():
        # 获取电影标题
        title = row["original_title"]
        # 如果标题无法匹配正则表达式模式（可能包含非法字符或格式不正确），则跳过当前行
        if not pattern.fullmatch(title):
            continue
        # 为标题添加双引号，确保格式统一
        title = "\"" + title + "\""
        # 遍历genres列表，写入标题和类型信息
        for g in row["genres"]:
            genre = g["name"]
            # 为类型添加双引号，确保格式统一
            genre = "\"" + genre + "\""
            out_genre.write(f"{title},{genre}\n")
        # 遍历keywords列表，写入标题和关键词信息
        for g in row["keywords"]:
            keyword = g["name"]
            # 为关键词添加双引号，确保格式统一
            keyword = "\"" + keyword + "\""
            out_keyword.write(f"{title},{keyword}\n")
        # 遍历production_companies列表，写入标题和制作人信息
        for g in row["production_companies"]:
            productor = g["name"]
            # 为制作人添加双引号，确保格式统一
            productor = "\"" + productor + "\""
            out_productor.write(f"{title},{productor}\n")

# 主程序入口
if __name__ == "__main__":
    # 调用Netflix函数进行数据预处理
    Netflix()
    # 调用TMDB函数进行数据预处理
    TMDB()
