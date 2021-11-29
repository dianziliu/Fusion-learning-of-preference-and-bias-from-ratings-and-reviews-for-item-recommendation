import pandas as pd
import datetime
def write2log(model_name,flags,res:pd.DataFame ,path):
    # 1. 写入模型名
    # 2. 写入模型运行参数
    with open(path,"a") as f:
    
        f.write(model_name+"\n")
        # 打印参数
    # 3. 写入所以结果
    res.to_csv(path)
