"""
version2.0
修改了doc_level_att 的部分内容
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as K_layers
from tensorflow.keras.layers import (Conv1D, Dense, Embedding, Input,
                                     concatenate)
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras import Model, callbacks
from  untils.gpu_config import *

TEST_FLAG=1

WordPre=1
ReviewPre=2
InterPre=4
RatingPre=8


def wrapper(func):
    """
    一个装饰器，用于进行一些运行测试和说明
    """
    def inner(*args,**kwargs):
        res=func(*args,**kwargs)
        # print("[!] build module {} OK".format(sys._getframe().f_code.co_name))
        if TEST_FLAG:
            print("[!] build module {} OK".format(func.__name__),end="\t")
            try:
                print(res.shape)
                # if res and res.hasattr("shape"):
                #     print(res.shape)
                # else:
                #     print()
            except:
                print()
        return res
    return inner



class PreferenceModel():
    """
    基本的基于文本的推荐系统模型
    假设P(w|u)和P(w|i)同属一个语义空间
    """
    name="PreModel"

    def __init__(self,flags,data_loader):

        self.num_class  = flags.num_class
        self.emb_size   = flags.emb_size
        self.batch_size = flags.batch_size
        self.epochs     = flags.epoch
        self.mode       = flags.mode # 用来控制偏好的影响
        self.doc_layers = flags.doc_layers
        self.doc_keep   = 1-flags.doc_dropout

        self.vocab_size = data_loader.vocab_size
        self.num_user   = data_loader.num_user
        self.num_item   = data_loader.num_item
        self.t_num      = data_loader.t_num
        self.maxlen     = data_loader.maxlen
        self.data_size  = data_loader.data_size+1
        self.vec_texts  = data_loader.vec_texts # 151255, 60
        
        self.ckpt_dir=os.path.join(flags.ckpt_dir,self.name)    
        
    def get_model(self):
        """
        写成显示的数据流会更好分析
        """

        # 获取输入
        u_input, i_input, text, utext, itext = self.init_input()

        # 构建v,d的相关结构
        self.build_word_level_layers()
        self.build_document_level_layers()

        # 构造基本的u|i 隐变量
        u_laten, i_laten = self.build_ui_laten(u_input, i_input)

        # !!! 需要拆分和细化
        # 获取u,i的文本特征 6,100|
        # 这里使用到了u_laten, i_laten
        docs_u, doc_u = self.get_w_u(utext)
        docs_i, doc_i = self.get_w_i(itext)

        # 基于注意力的w_u,w_i
        w_u, w_i = self.get_doc_level_att(doc_u, docs_u, doc_i, docs_i)
        # 预测的w
        pred_d = self.predict_value_d(u_input, w_u, i_input, w_i)
        
        r_w = self.predict_by_d(pred_d,u_input,i_input)

        model = Model(inputs=[self.u_input, self.i_input, self.text, self.utext, self.itext],
                      outputs=[r_w])

        self.model = model

    def init_input(self):

        self.u_input   = Input(name="u_input", dtype=tf.int32, shape=())
        self.i_input   = Input(name="i_input", dtype=tf.int32, shape=())
        self.text      = Input(name="text", dtype=tf.int32, shape=())
        self.utext     = Input(name="utext", dtype=tf.int32, shape=self.t_num)
        self.itext     = Input(name="itext", dtype=tf.int32, shape=self.t_num)
        self.keep_prob = 0.5
        # self.keep_prob  = Input(name="keep_prob", dtype=tf.float32)
        
        return self.u_input,self.i_input,self.text,self.utext,self.itext
    
    def build_ui_laten(self,uId,iId):
        """
        构建u和i的隐向量
        """
        # 交互级别
        self.u_embed=Embedding(self.num_user,self.emb_size,name="u_emb")
        self.i_embed=Embedding(self.num_item,self.emb_size,name="i_emb")

        self.u_latent=self.u_embed(uId)
        self.i_latent=self.i_embed(iId)
        

        # 评论级别
        self.u_review=Embedding(self.num_user,self.emb_size,name="u_review")
        self.i_review=Embedding(self.num_item,self.emb_size,name="i_review")

        self.u_review_latent=self.u_review(uId)
        self.i_review_latent=self.i_review(iId)

        # 评分级别
        self.u_rating=Embedding(self.num_user,self.emb_size,name="u_rating")
        self.i_rating=Embedding(self.num_item,self.emb_size,name="i_rating")

        self.u_rating_latent=self.u_rating(uId)
        self.i_rating_latent=self.i_rating(iId)

 
        return self.u_latent,self.i_latent


    def build_word_level_layers(self):
        # 构造vocab
        self.build_vocab()              # 构造v
        self.build_vocab_u()            # 构造P(v|u)
        self.build_vocab_i()            # 构造P(v|i)
        
    def build_document_level_layers(self):
        # 构造d
        self.build_document_latent()    # 构造P(d)
        self.build_document_user()      # 构造P(d|u)
        self.build_document_item()      # 构造P(d|i)
        self.budild_r_d()               # 构造P(r|d)
    
    @wrapper
    def build_vocab(self):
        """
        对字符进行词嵌入等操作，vec_texts 存储了通过其他模型预训练的词向量
        ？如何表示一个评论
        任务：调研如何通过词向量、或者之间处理整篇文章来进行表示
        """
        # assert self.emb_size==self.vec_texts.shape[1],"shape not same"
        
        # 加载词嵌入-这里使用了glove
        self.vec_embed = Embedding(self.vec_texts.shape[0],self.vec_texts.shape[1], name="build_w",
                                 trainable=False, weights=[self.vec_texts])
        
        # 1. 这里是个点
        self.w_embed=Embedding(self.vocab_size,self.emb_size,name="w_emb")
        self.w_att = Dense(self.emb_size*2, activation = 'tanh')
       
        # 
        
        layers=[(128,3),(128,5),(self.emb_size,3)]


        self.convs=[]
        for a, b in layers:
            self.convs.append(Conv1D(a, b, padding="same"))

        # self.conv1=Conv1D(128,3,padding="same")
        # self.conv2=Conv1D(128,5,padding="same")
        # self.conv3=Conv1D(self.emb_size,3,padding="same")


        # 问题是，如何构建语义向量空间

    @wrapper
    def build_vocab_u(self):
        """
        通过用户的历史记录刻画P(w|u)
        """
        self.wu_embed=Embedding(self.num_user,self.emb_size,name="wu_embed")

    @wrapper
    def build_vocab_i(self):
        """
        通过用户的历史记录刻画P(w|i)
        """
        self.wi_embed=Embedding(self.num_item,self.emb_size,name="wi_embed")

    def build_document_latent(self): 
        """
        定义计算得到document的结构
        """
        pass

    def build_document_user(self):
        """
        定义计算用户先验的document的结构，即P(d|u)
        """
        pass  
    def build_document_item(self):
        """
        定义计算物品先验的document的结构，即P(d|i)
        """
        pass  

    def get_document_user(self):
        pass  
    def get_document_item(self):
        pass  

    @wrapper
    def budild_r_d(self):
        """
        通过词空间预测评分
        """
        self.rd_layers=[]
        # for one_dim in layers:
        for i in [3,2,1]:
            one_dim=i*self.emb_size
            layer=Dense(one_dim,activation="elu",name="Pr_w_{}".format(one_dim))
            self.rd_layers.append(layer)
        Pred = Dense(1, bias_initializer=tf.constant_initializer(2),
                     activation="elu", name="P_r_w")
        self.rd_layers.append(Pred)
        

    def get_wui_convs(self,latent):
        latent=tf.reshape(latent,[-1,self.maxlen,self.emb_size])

        # for conv in self.convs[:-1]:
        #     latent=conv(latent)
        conv1=self.convs[0](latent)
        conv2=self.convs[1](conv1)

        hidden = tf.nn.relu(tf.concat([conv1,conv2], axis=-1))
        hidden = tf.nn.dropout(hidden, self.keep_prob)

        conv3 = tf.nn.relu(self.convs[2](hidden))
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        return conv3

    
    def get_certainty(self,alpha): # ?,6,60
        # alpha_sort = tf.sort(alpha, axis=-1)
        alpha_mean = tf.reduce_mean(alpha, axis=-1, keepdims = True) # ?,6,1
        # alpha_sort = alpha
        upper_mask = alpha>alpha_mean
        upper_mask = tf.cast(upper_mask, tf.float32)
        lower_mask = 1.-upper_mask   # ?,6,60

        alpha_lower = tf.reduce_mean(alpha*lower_mask, axis=-1, keepdims = True) # ?,6,1
        alpha_upper = tf.reduce_mean(alpha*upper_mask, axis=-1, keepdims = True)

        certainty = tf.nn.sigmoid((alpha_upper-alpha_mean)*(alpha_mean-alpha_lower))
        certainty = 2*certainty - 1
        # certainty = tf.expand_dims(certainty, axis=-1)
        return certainty

    @wrapper
    def get_word_level_att(self,uit_cnn, u_latent, i_latent, name='user'):

        uit_cnn_rsh = tf.reshape(uit_cnn, [-1, self.t_num, self.maxlen, self.emb_size])

        trans = self.w_att(uit_cnn_rsh) #[?,8,60,200]

        ui_latent = tf.concat([u_latent, i_latent], axis=-1)
        latent = tf.expand_dims(tf.expand_dims(ui_latent,1),1) #[?,1,1,200]
        alpha = tf.reduce_sum(trans*latent,axis=-1) #[?,8,60]


        alpha = tf.nn.softmax(alpha, axis=-1)
        if name == 'user':
            self.word_user_alpha = alpha
        else:
            self.word_item_alpha = alpha
        certainty = self.get_certainty(alpha)
        self.certainty = certainty

        alpha = tf.expand_dims(alpha, axis=-1) #[?,8,60,1]

        hidden = tf.reduce_sum(alpha*uit_cnn_rsh, axis=2) #[?,8,100]

        # print(certainty.shape, alpha.shape)

        return hidden #*certainty        

    @wrapper
    def doc_level_att(self, vec_1, vec_2, layer, name='user'):

        dist = tf.reduce_mean(tf.square(vec_1 - vec_2), axis=-1)*(layer+1)*10
        dist = -dist
        if layer == 0:
            self.vec_1 = vec_1
            self.vec_2 = vec_2
        alpha_1 = tf.nn.softmax(dist, axis=-1) # ?,6
        alpha_2 = tf.expand_dims(alpha_1, axis=-1)
        # if name == 'user':
        #     self.doc_user_alpha.append(alpha_1)
        # else:
        #     self.doc_item_alpha.append(alpha_1)

        return tf.reduce_sum(alpha_2*vec_2, axis=1, keepdims = True)


    @wrapper
    def get_doc_level_att(self, doc_user,docs_user, doc_item,docs_item):

        layers = self.doc_layers
        doc_att_layers=[]
        for i in range(layers):
            if i==0:
                i_temp = self.doc_level_att(doc_user, docs_item, i, 'item')
                u_temp = self.doc_level_att(doc_item, docs_user, i, 'user')
            else:
                i_temp = self.doc_keep* i_temp+ (1-self.doc_keep)*  self.doc_level_att(doc_user, docs_item, i, 'item')
                u_temp =  self.doc_keep* u_temp+(1-self.doc_keep)*  self.doc_level_att(doc_item, docs_user, i, 'user')                
            # self.doc_item.append(i_temp)
            # self.doc_user.append(u_temp)
        u_temp=tf.squeeze(u_temp,axis=1)
        i_temp=tf.squeeze(i_temp,axis=1)
        return u_temp,i_temp
    
    # get 方法似乎没有必要
    def get_w(index):
        """
        return w
        """

    @wrapper
    def get_w_u_i(self,uitext,name="user"):
        """
        计算P(w|u)和P(w|i)的通用模板
        """
        # 加载用户评论
        uidocs=self.vec_embed(uitext)
        # 计算用户对词空间的偏好

        uiw_emb=self.w_embed(uidocs)
        ui_cnn=self.get_wui_convs(uiw_emb)

        # !!! 这里需要进一步细化
        ui_watt=self.get_word_level_att(ui_cnn,self.u_latent,self.i_latent,name)
        doc_ui = tf.reduce_mean(ui_watt, axis=1, keepdims = True)

        return ui_watt,doc_ui        
    
    def get_w_u(self,utext):
        """
        return P(w|u)
        """
        return self.get_w_u_i(utext,"user")
        

    def get_w_i(self,itext):
        """
        return P(w|i)
        """
        # 加载物品评论
        return self.get_w_u_i(itext,"item")

    def predict_value_d(self,uId,w_u,iId,w_i):
        
        """
        直接P(w|u)和P(w|i)得到P(w|u,i)
        预测用户u对i的评论w
        return value of w
        """
        # 交互级偏置
        wu_embed=self.wu_embed(uId)
        wi_embed=self.wi_embed(iId)
        if self.mode & InterPre ==InterPre:
            laten=concatenate([wu_embed,w_u,wi_embed,w_i])
        else:
            laten=concatenate([w_u,w_i])
        layer=laten
        for i in [3,2,1]:
            one_dim=i*self.emb_size
            layer=Dense(one_dim,activation="relu",name="predcit_w_{}".format(one_dim))(layer)
        
        return layer


    def predict_by_d(self,w,uId,iId):
        """
        return P(r|w)
        """

        if self.mode & RatingPre ==RatingPre:
            w=concatenate([w,self.u_rating_latent,self.i_rating_latent])
        layer=w
        for one_layer in self.rd_layers:
            layer=one_layer(layer)
        return layer
        # for one_dim in layers:
        # for i in [3,2,1]:
        #     one_dim=i*self.emb_size
        #     layer=Dense(one_dim,activation="relu",name="Pr_w_{}".format(one_dim))(layer)
        
        # Pred = Dense(1,bias_initializer=tf.constant_initializer(3),name="P_r_w")(layer)
        # return Pred

    def predict_gmf(self,u_latent,i_latent,layers):
        layer=concatenate([u_latent,i_latent])
        for i,one in enumerate(layers):
            layer=Dense(one,activation="relu",name="GMF_{}_{}".format(i,one))(layer)
        layer=Dense(1,activation="relu",name="P_r_ui")(layer)
        return layer
    def final_pred(self,w,ulatent,ilatent,layers):
        """
        P(r|u,i)=P(r|w) *P(w|u,i)
        """
        layer = concatenate([w,ulatent,ilatent])
        for i,one in enumerate(layers):
            layer=Dense(one,activation="relu",name="final_pred_{}_{}".format(i,one))(layer)

        pred = Dense(1, name="P_r_uiw")(layer)

        return pred

    def train(self,data_loader):
        
        # 存储路径
        checkpoint_dir=self.ckpt_dir
        checkpoint_path=os.path.join(checkpoint_dir,"{}.h5".format(self.name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 通过回调进行保存
        cp_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="mean_absolute_error",
            save_best_only=True, verbose=1, save_weights_only=True, period=1)
        # 配置优化器
        # self.model.compile(optimizer="adam",loss="mean_squared_error",
        #     metrics = [RootMeanSquaredError(), "mean_absolute_error"])
        self.model.compile(optimizer="adam",loss="mean_squared_error",
            metrics = [RootMeanSquaredError(), "mean_absolute_error"])

        # 读取训练数据
        u_input, i_input, label, utext, itext, text = data_loader.all_train_data()


        train_data = {"u_input": u_input,
                      "i_input": i_input,
                      "text": text,
                      "utext": utext,
                      "itext": itext
                      }

        v_u_input, v_i_input, v_label, v_utext, v_itext, v_text=data_loader.eval()
        # valid_data = {"u_input": v_u_input,
        #               "i_input": v_i_input,
        #               "text": v_text,
        #               "utext": v_utext,
        #               "itext": v_itext
        #               }

        valid_data=[v_u_input, v_i_input, v_text, v_utext, v_itext ]
        valid=(valid_data,v_label)
        
        # 训练模型
        # self.model.summary()

        history = self.model.fit(train_data, label, epochs=self.epochs, verbose=1,
                                 callbacks=[cp_callback],validation_data=valid, validation_freq=1)
        # 返回训练的历史评价结果
        # 

        return history.history

if __name__=="__main__":
    print("hello,this is a work of RS")

