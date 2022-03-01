import pandas as pd, numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pymrmr
import keras
cancer_type =  ['BRCA']
index = 0
path = '/Data/nikhilanand_1921cs24/VAE_SVM/'+cancer_type[index]+'/mRNA'
mRNA_file_name = 'TCGA-BRCA.htseq_fpkm-uq_discrete.csv'

# transposing raw mRNA data and sorting them by patietnt Ids or TCGA Ids
# =============================================================================
df_mRNA = pd.read_csv(os.path.join(path,mRNA_file_name), header=None,index_col=0, delimiter=",", low_memory=False).T# read the csv data file and transpose it
df_mRNA.drop_duplicates(subset ="Ensembl_ID",keep='first',inplace=True)
df_mRNA=df_mRNA.sort_values(by=['Ensembl_ID'],ascending=True) # sorting based on tcga ids
# print(df_mRNA)
# =============================================================================
#Mapping mRNA patient ids with survival labels
# =============================================================================
df_mRNA_id=df_mRNA['Ensembl_ID'] # extracting tcga ids from mRNA dataframe
df_survival_label = pd.read_csv(os.path.join(path,'5_year_survival.csv'),delimiter=',') # reading survival csv file
survival_df = df_survival_label[df_survival_label['submitter_id.samples'].isin(df_mRNA_id)] # selcting survival label of tcga ids matching with mRNA tcga ids
survival_df=survival_df.sort_values(by=['submitter_id.samples'],ascending=True) # sorting survival labels based on tcga ids
survival_df.drop_duplicates(subset ="submitter_id.samples",keep='first',inplace=True)
survival_id = survival_df['submitter_id.samples'] # extracting tcga ids from survival labels
df_mRNA = df_mRNA[df_mRNA['Ensembl_ID'].isin(survival_id)] # selecting mRNA data of patients matching with avivalable survival ids
class_labels = survival_df['5_year_cutoff'] # fetching the class labels
# df_mRNA.insert(loc = len(df_mRNA.columns),column = 'label',value = class_labels.values)
X = df_mRNA.drop(df_mRNA.columns[0], axis=1)
X = X.astype(float).values
Y = class_labels.values

# print(X)
# print(Y.shape)


original_dim = X.shape[1]
latent_dim = 64

LOSS_THRESHOLD = 0.01

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('loss') < LOSS_THRESHOLD):
			print("\nReached %2.2f%% loss, so stopping training!!" %(LOSS_THRESHOLD*100))
			self.model.stop_training = True

# Instantiate a callback object
callbacks = myCallback()
# initializer_k0 = keras.initializers. RandomUniform(minval = -2, maxval =2)
# initializer_k1 = keras.initializers. RandomUniform(minval = -2, maxval =2)
# class Ada_act(keras.layers.Layer):
#     def __init__(self):
#         super(Ada_act, self).__init__()
#         self.k0 = self.add_weight(name='k0', shape = (), initializer=initializer_k0, trainable=True)
#         self.k1 = self.add_weight(name='k1', shape = (), initializer=initializer_k1, trainable=True)
        
        
#     def call(self, inputs):
#         return tf.maximum(self.k0, self.k0 + tf.multiply(inputs, self.k1))
# # Instantiate a Ada_act object

# Ada_act_1 = Ada_act()
# Ada_act_2 = Ada_act()
# Ada_act_3 = Ada_act()
# Ada_act_4 = Ada_act()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(2048, activation="tanh")(original_inputs)
x = layers.Dense(1024, activation="tanh")(x)
x = layers.Dense(512, activation="tanh")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(512, activation="tanh")(latent_inputs)
x = layers.Dense(1024, activation="tanh")(x)
x = layers.Dense(2048, activation="tanh")(x)
outputs = layers.Dense(original_dim)(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
import tensorflow_addons as tfa
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tfa.optimizers.Lookahead(optimizer)
vae.compile(optimizer, loss=tf.keras.losses.LogCosh())
vae.fit(X,X, epochs=50, batch_size =4,callbacks=[callbacks])
latent_features =  encoder.predict(X)
columns =[]
for i in range(1,latent_features.shape[1]+1):
    temp = 'mRNA_vae_'+str(i)
    columns.append(temp)
latent_df = pd.DataFrame(latent_features,columns = columns)
latent_df.insert(loc = 0,column = 'submitter_id.samples',value = df_mRNA['Ensembl_ID'].values)
latent_df.insert(loc = len(latent_df.columns),column = 'label',value = Y)
latent_df.to_csv(os.path.join(path,'vae_features_mRNA.csv'),index=False,header=True)








