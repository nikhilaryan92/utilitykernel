import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os,math,pickle
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate,cross_val_predict,StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_auc_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


os.environ["CUDA_VISIBLE_DEVICES"]="1"

#change the path to your local directory where TCGA-BRCA data is stored
path = '/Data/nikhilanand_1921cs24/TCGA/PyHIST/data_multimodal/'

# hyper-parameters of the utility kernel
c = 100
K_0 = 0.1
K_1 = 10
Alpha = 5

# number of PCA features by fixing the variance of 95%
NumComponents = 0.95

# number of samples of minority classes at various survival cut-offs
long_data_num_5yr = 204
long_data_num_6yr = 584
long_data_num_7yr = 187
long_data_num_8yr = 127
long_data_num_9yr = 102

# various survival cut-offs csv files
survival_file_name_5yr = 'class_5_year.csv'
survival_file_name_6yr = 'class_6_year.csv'
survival_file_name_7yr = 'class_7_year.csv'
survival_file_name_8yr = 'class_8_year.csv'
survival_file_name_9yr = 'class_9_year.csv'

#change for each survival year cutoff
long_data_num = long_data_num_6yr
survival_file_name = survival_file_name_6yr


def data_sampler(long_data,short_data):
	#subsampling the long data
	df = pd.DataFrame(long_data)
	long_data=df.sample(n = long_data_num)

	#generating balanced class data for training
	data = np.vstack((long_data,short_data))
	#print(data.shape)
	return data



#define utility kernel
def utility_kernel(X, Y):
    k_0 =K_0
    k_1 = K_1
    alpha = Alpha
    kernel_value = k_0+np.multiply(k_1,np.power((np.dot(X, Y.T)),alpha))
    return kernel_value


def my_pca(data,numComponents):
	#Applying PCA with 95% explained_variance_ratio_
	plt.style.use('ggplot')
	pca = PCA(n_components=numComponents, random_state=42)
	pca.fit(data) 
	data = pca.transform(data)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('number of components')
	plt.ylabel('Explained variance')
	plt.savefig(os.path.join(path,'elbow_plot_multimodal.png'),dpi=100)
	return data


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

#Loding and early fusion of multimodal data of TCGA-BRCA
CLN = np.loadtxt(os.path.join(path,'tcga_cln_final.csv'), delimiter=",")
WSI = np.loadtxt(os.path.join(path,'tcga_allWSIpatchfused.csv'), delimiter=",")
EXP = np.loadtxt(os.path.join(path,'tcga_exp_final.csv'), delimiter=",")
CNV = np.loadtxt(os.path.join(path,'tcga_cnv_final.csv'), delimiter=",")
CLN_EXP    = np.concatenate((CLN,EXP),axis=1)
CLN_CNV    = np.concatenate((CLN,CNV),axis=1)
CLN_WSI    = np.concatenate((CLN,WSI),axis=1)
EXP_CNV    = np.concatenate((EXP,CNV),axis=1)
EXP_WSI    = np.concatenate((EXP,WSI),axis=1)
CNV_WSI    = np.concatenate((CNV,WSI),axis=1)
CLN_EXP_CNV     = np.concatenate((CLN,EXP,CNV),axis=1)
CLN_EXP_WSI     = np.concatenate((CLN,EXP,WSI),axis=1)
EXP_CNV_WSI     = np.concatenate((EXP,CNV,WSI),axis=1)
CNV_CLN_WSI     = np.concatenate((CNV,CLN,WSI),axis=1)
multimodal     = np.concatenate((WSI,CLN,CNV,EXP),axis=1)
y_multimodal 	  = np.loadtxt(os.path.join(path, survival_file_name))



def model_run(X,filename,final_df):
	result_file = "".join([filename, '_results.csv'])
	print('Runing', result_file)
	X_multimodal=my_pca(X,NumComponents)
	'''
	df_X_multimodal=pd.DataFrame(X_multimodal)
	X_multimodal_cov=df_X_multimodal.corr(method ='pearson')
	X_multimodal_cov.to_csv(os.path.join(path,'covariance_multimodal.csv'))
	'''
	# arranging data classwise
	count_short =0
	count_long = 0
	for index in range(0,X_multimodal.shape[0]):
		if(y_multimodal[index]==1 and count_short ==0):
			X_short = X_multimodal[index]
			Y_short = y_multimodal[index]
			count_short = count_short+1
		elif(y_multimodal[index]==1 and count_short !=0):
			X_short = np.vstack((X_short,X_multimodal[index]))
			Y_short = np.vstack((Y_short,y_multimodal[index]))
		elif(y_multimodal[index]==0 and count_long==0):
			X_long = X_multimodal[index]
			Y_long = y_multimodal[index]
			count_long = count_long+1
		elif(y_multimodal[index]==0 and count_long!=0):
			X_long = np.vstack((X_long,X_multimodal[index]))
			Y_long = np.vstack((Y_long,y_multimodal[index]))

	# concatenating class labels with their corresponding features
	short_data=np.hstack((X_short,Y_short))
	long_data=np.hstack((X_long,Y_long))


	# fit a SVM model to the data
	model = SVC(C=c,kernel=utility_kernel)
	#model = SVC(C=10,kernel='sigmoid',degree=5)
	#model = GaussianNB()
	#model = tree.DecisionTreeClassifier()
	#model = RandomForestClassifier(max_depth=None, random_state=0,min_samples_split=0.5)
	aucs_test = []
	aucs_train = []
	roc_auc_new =0

	filepath = './svm.sav'
	for itr in range(0,1): 
		data=data_sampler(long_data,short_data)
		X = data[:,0:data.shape[1]-1]
		y = data[:,data.shape[1]-1]
		
		skf = StratifiedKFold(n_splits=10)
		for train_index, test_index in skf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			model=model.fit(X_train, y_train)
			y_train_pred=model.predict(X_train)
			roc_auc_train=roc_auc_score(y_train, y_train_pred)
			aucs_train.append(roc_auc_train)
			y_pred=model.predict(X_test)
			roc_auc=roc_auc_score(y_test, y_pred)
			aucs_test.append(roc_auc)
			if(roc_auc>roc_auc_new):
				pickle.dump(model, open(filepath, 'wb'))
				roc_auc_new=roc_auc
			model = pickle.load(open(filepath, 'rb'))
	final_model = pickle.load(open(filepath, 'rb'))
	y_pred_multimodal=final_model.predict(X_multimodal)
	tn_fp_fn_tp = confusion_matrix(y_multimodal,y_pred_multimodal).ravel()
	roc=roc_auc_score(y_multimodal,y_pred_multimodal)
	print(tn_fp_fn_tp)
	print(roc)
	results = np.append(roc,tn_fp_fn_tp)

	results_csv = open(result_file, "a")

	# loop over the class labels and extracted features
	vec = ",".join([str(v) for v in results])
	results_csv.write("{}\n".format(vec))
	# close the CSV file
	results_csv.close()

	output = y_pred_multimodal
	#print(output)
	# convert array into dataframe
	DF = pd.DataFrame(output)
	final_df = pd.concat([final_df,DF],axis=1)
	print(final_df.shape) 

	return final_df


final_df = pd.DataFrame(y_multimodal) # initializing the 1st column of final_df with true class labels
for i in range(1,16):
	# uni-modal
	if(i==1):
		final_df= model_run(CLN,'CLN',final_df)
	if(i==2):
		final_df= model_run(WSI,'WSI',final_df)
	if(i==3):
		final_df= model_run(EXP,'EXP',final_df)
	if(i==4):
		final_df= model_run(CNV,'CNV',final_df)
	# bi-modal
	if(i==5):
		final_df= model_run(CLN_EXP,'CLN_EXP',final_df)
	if(i==6):
		final_df= model_run(CLN_CNV,'CLN_CNV',final_df)
	if(i==7):
		final_df= model_run(CLN_WSI,'CLN_WSI',final_df)
	if(i==8):
		final_df= model_run(EXP_CNV,'EXP_CNV',final_df)
	if(i==9):
		final_df= model_run(EXP_WSI,'EXP_WSI',final_df)	
	if(i==10):
		final_df= model_run(CNV_WSI,'CNV_WSI',final_df)
	# tri-modal
	if(i==11):
		final_df= model_run(CLN_EXP_CNV,'CLN_EXP_CNV',final_df)
	if(i==12):
		final_df= model_run(CLN_EXP_WSI,'CLN_EXP_WSI',final_df)
	if(i==13):
		final_df= model_run(EXP_CNV_WSI,'EXP_CNV_WSI',final_df)
	if(i==14):
		final_df= model_run(CNV_CLN_WSI,'CNV_CLN_WSI',final_df)
	# quad-modal
	if(i==15):
		final_df= model_run(multimodal,'multimodal',final_df)

# save the dataframe as a csv file
final_df.to_csv('output.csv', header=False, index=False)
