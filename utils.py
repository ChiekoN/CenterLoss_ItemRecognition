import os
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
from tensorboard.plugins import projector
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


image_dir = "image_data"

class tbProjector(tf.keras.callbacks.Callback):

    def __init__(self, embedding_model, x_test, y_test, log_dir, metadata):
        super(tbProjector, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.log_dir = log_dir
        self.metadata = metadata
        self.output()

    def output(self):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        test_emb_tensor = tf.Variable(x_test_embeddings)
        checkpoint = tf.train.Checkpoint(embedding=test_emb_tensor)
        checkpoint.save(os.path.join(self.log_dir, "embedding.ckpt"))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = self.metadata
        projector.visualize_embeddings(self.log_dir, config)

    def on_epoch_end(self, epoch, logs=None):
        self.output()



class PCAPlotter(tf.keras.callbacks.Callback):

    def  __init__(self, model, feature_model, x_test, y_test, epoch, lambda_cl):
        super(PCAPlotter, self).__init__()
        self.model = model
        self.feature_model = feature_model
        self.x_test = x_test
        self.y_test = y_test
        self.y_seq = np.array(list(set(y_test))) # Sequence of unique labels
        self.fig = plt.figure(figsize=(9, 4))
        #self.ax = plt.subplot(1, 2, 1)
        self.epoch = epoch
        self.lambda_cl = lambda_cl
        plt.ion()
        
        self.save_path = 'results'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        print('y_seq = {}, len = {}'.format(self.y_seq, self.y_seq.shape))

    def visualize_train(self, feat, centers, cur_epoch):
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
            '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.figure()
        plt.clf()
        for i in range(10):
            plt.plot(feat[self.y_test == self.y_seq[i], 0], feat[self.y_test == self.y_seq[i], 1], '.', c=c[i])
        plt.plot(centers[self.y_seq, 0], centers[self.y_seq, 1], 'kx', mew=2, ms=4)
        plt.title('Training data. Lambda_centerloss = {}, Epoch = {}/{}'.format(self.lambda_cl, cur_epoch, self.epoch))
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.savefig('{}/train_plot.png'.format(self.save_path))
        plt.close()

    def plot_feature_2D(self, epoch):
        feat = self.feature_model.predict(self.x_test)
        centers = self.model.get_layer('centerlosslayer').get_weights()[0]
        pca = PCA(n_components=2)
        pca_feat = pca.fit_transform(feat)
        pca_centers = pca.fit_transform(centers)
        self.visualize_train(pca_feat, pca_centers, epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.plot_feature_2D(epoch)



def create_batch(batch_size=256, mode="train"):
    ''' Create batch of triplets.
        Randomly choose an image from a randomly chosen class as anchor,
        also choose the other image from the same class as positive,
        then randomly choose an image from randomly chose class except the class
        selected previously as negative.

        batch_size = Batch size
        mode = train / test (data directory)
        Return : list of [array of anchor images, array of positive images, array of negative images]
                 (each array contains batch_size data)
    '''
    rng = default_rng()

    x_anc = np.zeros((batch_size, 224, 224, 3))
    x_pos = np.zeros((batch_size, 224, 224, 3))
    x_neg = np.zeros((batch_size, 224, 224, 3))
    
    if mode == 'train':
        data_dir = os.path.join(image_dir, "train")
        
    elif mode == 'test':
        data_dir = os.path.join(image_dir, "test")
    
        
    class_list = sorted(os.listdir(data_dir))
    
    rand_classes = rng.choice(len(class_list), batch_size, replace=True)
    
    for i in range(batch_size):
        class_idx = rand_classes[i]
        anc_class_dir = os.path.join(data_dir, class_list[class_idx])
        anc_imglist = os.listdir(anc_class_dir)
        anc_pos_idx = rng.choice(len(anc_imglist), 2, replace=False)
        
        img_anc = tf.keras.preprocessing.image.load_img(os.path.join(anc_class_dir, anc_imglist[anc_pos_idx[0]]))
        img_anc = img_anc.resize((224, 224))
        img_anc = tf.keras.preprocessing.image.img_to_array(img_anc)
        
        img_pos = tf.keras.preprocessing.image.load_img(os.path.join(anc_class_dir, anc_imglist[anc_pos_idx[1]]))
        img_pos = img_pos.resize((224, 224))
        img_pos = tf.keras.preprocessing.image.img_to_array(img_pos)
        
        # Select negative class (!= class_idx)
        all_class_idx = np.arange(len(class_list))
        neg_mask = (all_class_idx != class_idx)
        
        all_neg_classes = all_class_idx[neg_mask]
        neg_class_idx = rng.choice(all_neg_classes)
        
        neg_class_dir = os.path.join(data_dir, class_list[neg_class_idx])
        neg_imglist = os.listdir(neg_class_dir)
        neg_idx = rng.choice(len(neg_imglist))
        
        img_neg = tf.keras.preprocessing.image.load_img(os.path.join(neg_class_dir, neg_imglist[neg_idx]))
        img_neg = img_neg.resize((224, 224))
        img_neg = tf.keras.preprocessing.image.img_to_array(img_neg)
        
        x_anc[i] = img_anc/255.
        x_pos[i] = img_pos/255.
        x_neg[i] = img_neg/255.
    
    return [x_anc, x_pos, x_neg]


def create_testdata(class_size=10, datadir='test', tblog_dir='.', metadatafile='metadata.tsv', seed=None):
    ''' Create test data. It's useable to see Tensorboard Embeddings Projectsion.
        This function also outputs 'metadata.tsv', which records each class name corresponding to
        each data point in return array.
        Choose classes of class_size randomly and get all image data from them to create test data.
        class_size = number of classes in test data
        datadir = test data directory
        tblog_dir = Tensorboard log directory
        metadatafile = file name of metadata file for Tensorboard
        Return : test data 
    '''

    rng = default_rng(seed)

    test_dir = os.path.join(image_dir, datadir)
    class_list_test = sorted(os.listdir(test_dir))
    
    test_data_list = []
    test_data_label = []
    
    all_class_size = len(class_list_test)
    if class_size > all_class_size:
        class_size = all_class_size
        
    class_list_test_rand = rng.choice(class_list_test, class_size, replace=False)
    
    for test_class in class_list_test_rand:
                      
        test_class_dir = os.path.join(test_dir, test_class)
        imglist = os.listdir(test_class_dir)
        
        for img_file in imglist:
                
            # Add this class to the list
            test_data_label.append(test_class)
            
            img = tf.keras.preprocessing.image.load_img(os.path.join(test_class_dir, img_file))
            img = img.resize((224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img /= 255.
            
            test_data_list.append(img)

            
    test_data = np.array(test_data_list)
    
    # Write metadata file
    with open(os.path.join(tblog_dir, metadatafile), "w") as f:
        for label in test_data_label:
            f.write("{}\n".format(label))
    
    return (test_data, test_data_label)



def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:, :emb_dim], y_pred[:, emb_dim:2*emb_dim], y_pred[:, 2*emb_dim:]
        dp = tf.reduce_mean(tf.square(anc - pos), axis=1)
        dn = tf.reduce_mean(tf.square(anc - neg), axis=1)
        return tf.maximum(dp - dn + alpha, 0.)
    return loss






