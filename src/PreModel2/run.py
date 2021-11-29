import os
import sys

sys.path.append(".")

import pandas as pd
import tensorflow as tf
# from tensorflow.compat.v1 import flags as tf_flags
from absl import flags as tf_flags

from src.PreModel2.PreModel import PreferenceModel
# from src.model import MultHTI
from untils import Data_Loader

if __name__=="__main__":
    
    # filename = os.path.join(prefix,'Grocery_and_Gourmet_Food_5.json')
    paths = [
            "Video_Games_5.json",
             "Sports_and_Outdoors_5.json",
             "Office_Products_5.json",
             'Grocery_and_Gourmet_Food_5.json',
             'Musical_Instruments_5.json'
             ]
    prefix = 'amazon_data'
    # filename = os.path.join(prefix,paths[0])
    filename=paths[0]
    flags = tf_flags.FLAGS 	
    tf_flags.DEFINE_string('filename', filename, 'name of file')
    tf_flags.DEFINE_string("res_dir", filename, "name of dir to store result")
    tf_flags.DEFINE_integer('batch_size', 128, 'batch size')
    tf_flags.DEFINE_integer('emb_size', 100, 'embedding size')
    tf_flags.DEFINE_integer('num_class', 5, "num of classes")
    tf_flags.DEFINE_integer('epoch', 20, 'epochs for training')
    tf_flags.DEFINE_string('ckpt_dir', os.path.join(
        "CKPT_DIR", "HTI_"+filename.split('.')[0]), 'directory of checkpoint')
    tf_flags.DEFINE_string('train_test', 'train', 'training or test')
    tf_flags.DEFINE_string("glovepath", "glove", "glove path")
    tf_flags.DEFINE_string("res_path", "res/res.csv", "save predict res")
    tf_flags.DEFINE_float('test_size', "0.2", "set test size to split data")
    tf_flags.DEFINE_string('res', "", "res path to save")
    tf_flags.DEFINE_integer('mode', 4, "12=4&8,4表示 inter pre, 8 表示 rating pre")
    tf_flags.DEFINE_integer('doc_layers', 3, "doc层注意力的层数")
    tf_flags.DEFINE_float('doc_dropout', .3, "doc层注意力的层数")

    # tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
    flags(sys.argv)
    flags.filename=os.path.join(prefix,flags.filename)
    data_loader = Data_Loader(flags)
    model=PreferenceModel(flags, data_loader)
    
    model.get_model()
    res=model.train(data_loader)

    res_Df=pd.DataFrame(res)

    print(res_Df)
    if flags.res!="":
        res_Df.to_csv(flags.res,index=False)
