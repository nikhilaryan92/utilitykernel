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
path = '/Data/nikhilanand_1921cs24/TCGA/PyHIST/data_multimodal/'

c = 1
K_0 = 0.1
K_1 = 10
Alpha = 5
NumComponents = 0.95
long_data_num = 187

def data_sampler(long_data,short_data):
	#subsampling the long data
	df = pd.DataFrame(long_data)
	long_data=df.sample(n = long_data_num)

	#generating balanced class data for training
	data = np.vstack((long_data,short_data))
	#print(data.shape)
	return data
#define custom kernel
def CobbD_kernel(X, Y):
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
X_CLN = np.loadtxt(os.path.join(path,'tcga_cln_final.csv'), delimiter=",")# Change the path to your local system
X_WSI = np.loadtxt(os.path.join(path,'tcga_allWSIpatchfused.csv'), delimiter=",")# Change the path to your local system
X_EXP = np.loadtxt(os.path.join(path,'tcga_exp_final.csv'), delimiter=",")# Change the path to your local system
X_CNV = np.loadtxt(os.path.join(path,'tcga_cnv_final.csv'), delimiter=",")# Change the path to your local system
X_CLN_EXP     = np.concatenate((X_CLN,X_EXP),axis=1)
X_CLN_CNV    = np.concatenate((X_CLN,X_CNV),axis=1)
X_CLN_WSI     = np.concatenate((X_CLN,X_WSI),axis=1)
X_EXP_CNV     = np.concatenate((X_EXP,X_CNV),axis=1)
X_EXP_WSI     = np.concatenate((X_EXP,X_WSI),axis=1)
X_CNV_WSI     = np.concatenate((X_CNV,X_WSI),axis=1)
X_CLN_EXP_CNV     = np.concatenate((X_CLN,X_EXP,X_CNV),axis=1)
X_CLN_EXP_WSI     = np.concatenate((X_CLN,X_EXP,X_WSI),axis=1)
X_EXP_CNV_WSI     = np.concatenate((X_EXP,X_CNV,X_WSI),axis=1)
X_CNV_CLN_WSI     = np.concatenate((X_CNV,X_CLN,X_WSI),axis=1)
X_multimodal     = np.concatenate((X_WSI,X_CLN,X_CNV,X_EXP),axis=1)
y_multimodal 	  = np.loadtxt(os.path.join(path,'class_7_year.csv'))

'''
#Loding and Late fusion of multimodal data of TCGA-BRCA
X_WSI = np.loadtxt(os.path.join(path,'allWSIpatchfused.csv'), delimiter=",")# Change the path to your local system
X_CLN = np.loadtxt(os.path.join(path,'gatedAtnClnOutput.csv'), delimiter=",")# Change the path to your local system
X_CNV = np.loadtxt(os.path.join(path,'gatedAtnCnvOutput.csv'), delimiter=",")# Change the path to your local system
X_EXP = np.loadtxt(os.path.join(path,'gatedAtnExpOutput.csv'), delimiter=",")# Change the path to your local system
y 	  = np.loadtxt(os.path.join(path,'class.csv'))
X_multimodal     = np.concatenate((X_WSI,X_CLN,X_CNV,X_EXP),axis=1)
'''


def model_run(X,filename):
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

	# concatenating class labels with the features
	short_data=np.hstack((X_short,Y_short))
	long_data=np.hstack((X_long,Y_long))


	# fit a SVM model to the data
	#model = SVC(C=c,kernel=CobbD_kernel)
	#model = SVC(C=10,kernel='sigmoid',degree=5)
	#model = GaussianNB()
	#model = tree.DecisionTreeClassifier()
	model = RandomForestClassifier(max_depth=None, random_state=0,min_samples_split=0.5)
	aucs = []
	aucs_train = []
	roc_auc_new =0

	filepath = './svm.sav'
	for itr in range(0,10):
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
			aucs.append(roc_auc)
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

	results_csv = open(os.path.join(path,result_file), "a")

	# loop over the class labels and extracted features
	vec = ",".join([str(v) for v in results])
	results_csv.write("{}\n".format(vec))
	# close the CSV file
	results_csv.close()

for i in range(1,16):
	if(i==1):
		model_run(X_CLN,'X_CLN')
	elif(i==2):
		model_run(X_WSI,'X_WSI')
	elif(i==3):
		model_run(X_EXP,'X_EXP')
	elif(i==4):
		model_run(X_CNV,'X_CNV')
	elif(i==5):
		model_run(X_CLN_EXP,'X_CLN_EXP')
	elif(i==6):
		model_run(X_CLN_CNV,'X_CLN_CNV')
	elif(i==7):
		model_run(X_CLN_WSI,'X_CLN_WSI')
	elif(i==8):
		model_run(X_EXP_CNV,'X_EXP_CNV')
	elif(i==9):
		model_run(X_EXP_WSI,'X_EXP_WSI')	
	elif(i==10):
		model_run(X_CNV_WSI,'X_CNV_WSI')
	elif(i==11):
		model_run(X_CLN_EXP_CNV,'X_CLN_EXP_CNV')
	elif(i==12):
		model_run(X_CLN_EXP_WSI,'X_CLN_EXP_WSI')
	elif(i==13):
		model_run(X_EXP_CNV_WSI,'X_EXP_CNV_WSI')
	elif(i==14):
		model_run(X_CNV_CLN_WSI,'X_CNV_CLN_WSI')
	elif(i==15):
		model_run(X_multimodal,'X_multimodal')

