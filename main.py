import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import time



# dataset link : https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
data=pd.read_csv("pima-indians-diabetes.csv",header=None)

data.columns=['preg','plas','pres','skin','test','mass','pedi','age','class']
X=data.iloc[:,1:8]
y=data['class']

##split the dataset


test_sizes=[0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.1,0.05]

train_scores =[]
test_scores = []
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=31)
    model_dt=DecisionTreeClassifier(random_state=37,min_samples_leaf=3,ccp_alpha=0.01)
    model_dt.fit(X_train, y_train)
    train_score=model_dt.score(X_train,y_train)
    train_scores.append(train_score)
    test_score = model_dt.score(X_test, y_test)
    test_scores.append(test_score)


fig, ax = plt.subplots()
ax.set_xlabel("training sizes")
ax.set_ylabel("accuracy")
ax.set_title("Decision Tree Accuracy Based on Training Size")
ax.plot([1-x for x in test_sizes], train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot([1-x for x in test_sizes], test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Decision Tree Model.png")

#base on the plot picking the alph=0.02

#KNN classfier


i = 0  # initialize plot counter


fig, axs = plt.subplots(3,2, figsize=(14, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()


test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
for i in range(len(test_sizes)):
    training_size=1-test_sizes[i]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    neightbour=[2,4,6,8,10,12,14]
    model_nbrs=[]
    for n in neightbour:
        model_nbr=KNeighborsClassifier(n_neighbors=n,n_jobs=2)
        model_nbr.fit(X_train,y_train)
        model_nbrs.append(model_nbr)

    train_scores = [model_nbr.score(X_train, y_train) for model_nbr in model_nbrs]
    test_scores = [model_nbr.score(X_test, y_test) for model_nbr in model_nbrs]

    axs[i].plot(neightbour, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    axs[i].plot(neightbour, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    axs[i].set_xlabel("n")
    axs[i].set_ylabel("accuracy")
    axs[i].set_title("KNN with training size %.1f" % training_size)
    axs[i].legend()
plt.savefig("KNN Model.png")

##Neural Network
train_scores=[]
test_scores=[]
test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
model_nnet=MLPClassifier(max_iter=1000)
for  i in range(len(test_sizes)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    parameters={'hidden_layer_sizes':[50,100,200,300,400],'alpha':[0.0001,0.01,0.05],'learning_rate_init':[0.001, 0.005]}
    model_nnets=GridSearchCV(model_nnet,parameters)
    model_nnets.fit(X_train,y_train)
    train_score=model_nnets.score(X_train, y_train)
    train_scores.append(train_score)
    test_score=model_nnets.score(X_test,y_test)
    test_scores.append(test_score)

fig, ax = plt.subplots()
ax.set_xlabel("training size")
ax.set_ylabel("accuracy")
ax.set_title("Neural Network Model Accuracy")
ax.plot([1-x for x in test_sizes], train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot([1-x for x in test_sizes], test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Neural Network Model.png")


#Boosting
train_scores=[]
test_scores=[]
test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
n_estimators=[]
parameters = {'n_estimators': [50, 100, 200, 300, 400, 500, 1000], 'ccp_alpha': [0.0001, 0.01, 0.05],
              'learning_rate': [0.001, 0.005]}

for  i in range(len(test_sizes)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    model_boosting=GradientBoostingClassifier(random_state=37,n_iter_no_change=100,max_features='auto')
    model_boostings=GridSearchCV(model_boosting,parameters)
    model_boostings.fit(X_train, y_train)
    train_score = model_boostings.score(X_train, y_train)
    train_scores.append(train_score)
    test_score = model_boostings.score(X_test, y_test)
    test_scores.append(test_score)


fig, ax = plt.subplots()
ax.set_xlabel("training size")
ax.set_ylabel("accuracy")
ax.set_title("Boosting Model Accuracy")
ax.plot([1-x for x in test_sizes], train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot([1-x for x in test_sizes], test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Boosting Model.png")


##SVM

fig, axs = plt.subplots(3,2, figsize=(14, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()


test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
for i in range(len(test_sizes)):
    training_size=1-test_sizes[i]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    kernals=['linear','poly','rbf','sigmoid']
    model_svms=[]
    for kernal in kernals:
        model_svm=SVC(kernel=kernal)
        model_svm.fit(X_train,y_train)
        model_svms.append(model_svm)

    train_scores = [model_svm.score(X_train, y_train) for model_svm in model_svms]
    test_scores = [model_svm.score(X_test, y_test) for model_svm in model_svms]

    axs[i].plot(kernals, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    axs[i].plot(kernals, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    axs[i].set_xlabel("n")
    axs[i].set_ylabel("accuracy")
    axs[i].set_title("SVM with training size %.1f" % training_size)
    axs[i].legend()
plt.savefig("SVM Model.png")

# measure the performance
models=[]
training_time=[]
test_time=[]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=31)
dt=DecisionTreeClassifier(random_state=37,min_samples_leaf=3,ccp_alpha=0.01)
models.append(dt)
nbr=KNeighborsClassifier(n_neighbors=8)
models.append(nbr)
svm=SVC(kernel='linear',class_weight='balanced')
models.append(svm)
nnet=MLPClassifier(max_iter=1000)
models.append(nnet)
boost=GradientBoostingClassifier(random_state=37,n_iter_no_change=100)
models.append(boost)

for model in models:
    start = time.time()
    model.fit(X_train,y_train)
    end=time.time()
    training_time.append(end - start)
    start = time.time()
    model.predict(X_test)
    end = time.time()
    test_time.append(end-start)
print(training_time)
print(test_time)

# ##2nd dataset
data2=pd.read_csv("Taiwan_data.csv",skiprows=1,header=None)
X=data2.iloc[:,1:]
y=data2.iloc[:,0]


test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]

train_scores =[]
test_scores = []
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=31)
    model_dt=DecisionTreeClassifier(random_state=37,min_samples_leaf=3,ccp_alpha=0.01,class_weight='balanced')
    model_dt.fit(X_train, y_train)
    train_score=f1_score(y_train,model_dt.predict(X_train),average='macro')
    train_scores.append(train_score)
    test_score=f1_score(y_test,model_dt.predict(X_test),average='macro')
    test_scores.append(test_score)


fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel("training sizes")
ax.set_ylabel("f1 score")
ax.set_title("Decision Tree F1 score Based on Training Size")
ax.plot([1-x for x in test_sizes], train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot([1-x for x in test_sizes], test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Decision Tree Model Taiwan.png")


#KNN model




fig, axs = plt.subplots(3,2, figsize=(14, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()


test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
for i in range(len(test_sizes)):
    train_scores = []
    test_scores = []
    training_size=1-test_sizes[i]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    neightbour=[2,4,6,8,10,12,14]
    model_nbrs=[]
    for n in neightbour:
        model_nbr=KNeighborsClassifier(n_neighbors=n,n_jobs=2)
        model_nbr.fit(X_train,y_train)
        train_score = f1_score(y_train, model_nbr.predict(X_train), average='macro')
        train_scores.append(train_score)
        test_score = f1_score(y_test, model_nbr.predict(X_test), average='macro')
        test_scores.append(test_score)

    axs[i].plot(neightbour, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    axs[i].plot(neightbour, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    axs[i].set_xlabel("n")
    axs[i].set_ylabel("f1 score")
    axs[i].set_title("KNN with training size %.1f" % training_size)
    axs[i].legend()
plt.savefig("KNN Model Taiwan.png")

#SVM model
fig, axs = plt.subplots(3,2, figsize=(14, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()
test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
for i in range(len(test_sizes)):
    train_scores = []
    test_scores = []
    training_size=1-test_sizes[i]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    kernals=['linear','poly','rbf','sigmoid']
    for kernal in kernals:
        model_svm=SVC(kernel=kernal,class_weight='balanced')
        model_svm.fit(X_train,y_train)
        train_score = f1_score(y_train, model_svm.predict(X_train), average='macro')
        train_scores.append(train_score)
        test_score = f1_score(y_test, model_svm.predict(X_test), average='macro')
        test_scores.append(test_score)
    axs[i].plot(kernals, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    axs[i].plot(kernals, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    axs[i].set_xlabel("kernels")
    axs[i].set_ylabel("f1 score")
    axs[i].set_title("SVM with training size %.1f" % training_size)
    axs[i].legend()
plt.savefig("SVM Model Taiwan.png")


##Neural Network
train_scores=[]
test_scores=[]
time=[]
test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
model_nnet=MLPClassifier(max_iter=1000)
for  i in range(len(test_sizes)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    parameters={'hidden_layer_sizes':[50,100,200,300,400],'alpha':[0.0001,0.01,0.05],'learning_rate_init':[0.001, 0.005]}
    model_nnets=GridSearchCV(model_nnet,parameters)
    model_nnets.fit(X_train,y_train)
    train_score = f1_score(y_train, model_nnets.predict(X_train), average='macro')
    train_scores.append(train_score)
    test_score = f1_score(y_test, model_nnets.predict(X_test), average='macro')
    test_scores.append(test_score)

fig, ax = plt.subplots()
ax.set_xlabel("training size")
ax.set_ylabel("accuracy")
ax.set_title("Neural Network Model F1 score")
ax.plot([1-x for x in test_sizes], train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot([1-x for x in test_sizes], test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Neural Network Model Taiwan.png")

print(time)

# Boosting
fig, axs = plt.subplots(3,2, figsize=(14, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()

time=[]

test_sizes=[0.6,0.5,0.4,0.3,0.2,0.1]
n_estimators=[]
parameters = {'n_estimators': [100, 200, 500, 800], 'ccp_alpha': [0.0001, 0.01 ,0.1]}
train_scores = []
test_scores = []

for  i in range(len(test_sizes)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sizes[i], random_state=31)
    model_boosting=GradientBoostingClassifier(random_state=32,n_iter_no_change=100)
    model_boostings=GridSearchCV(model_boosting,parameters)
    model_boostings.fit(X_train, y_train)
    train_score = f1_score(y_train, model_boostings.predict(X_train), average='macro')
    train_scores.append(train_score)
    test_score = f1_score(y_test, model_boostings.predict(X_test), average='macro')
    test_scores.append(test_score)

fig, ax = plt.subplots()
ax.set_xlabel("training size")
ax.set_ylabel("f1 score")
ax.set_title("Boosting Model F1 score")
ax.plot([1-x for x in test_sizes], train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot([1-x for x in test_sizes], test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Boosting Model Taiwan.png")


models=[]
training_time=[]
test_time=[]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=31)
dt=DecisionTreeClassifier(random_state=37,min_samples_leaf=3,ccp_alpha=0.01)
models.append(dt)
nbr=KNeighborsClassifier(n_neighbors=8)
models.append(nbr)
svm=SVC(kernel='linear',class_weight='balanced')
models.append(svm)
nnet=MLPClassifier(max_iter=1000,hidden_layer_sizes=500)
models.append(nnet)
boost=GradientBoostingClassifier(random_state=37,n_iter_no_change=100,n_estimators=500)
models.append(boost)

for model in models:
    start = time.time()
    model.fit(X_train,y_train)
    end=time.time()
    training_time.append(end - start)
    start = time.time()
    model.predict(X_test)
    end = time.time()
    test_time.append(end-start)
print(training_time)
print(test_time)
#
## test neural netowrk by layers


data2=pd.read_csv("Taiwan_data.csv",skiprows=1,header=None)
X=data2.iloc[:,1:]
y=data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)


training_time=[]
predict_time=[]
training_scores=[]
prediction_scores=[]
layers=[50,100,200,300,400,500,800,1000]
for layer in layers:
    mlp_model=MLPClassifier(hidden_layer_sizes=layer)
    start=time.time()
    mlp_model.fit(X_train,y_train)
    end=time.time()
    training_scores.append(f1_score(y_train,mlp_model.predict(X_train),average='macro'))
    mlp_model.predict(X_test)
    end2=time.time()
    prediction_scores.append(f1_score(y_test,mlp_model.predict(X_test),average='macro'))

    training_time.append(end-start)
    predict_time.append(end2-end)

fig, ax = plt.subplots()
ax.set_xlabel("Neutral Layer Sizes ")
ax.set_ylabel("Time Spent")
ax.set_title("Neural Network Training Time ")
ax.plot(layers, training_time, marker='o', label="training_time",
        drawstyle="steps-post")
ax.plot(layers , predict_time, marker='o', label="prediction_time",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Neutral Network Model Performance.png")

fig, ax = plt.subplots()
ax.set_xlabel("Neutral Layer Sizes ")
ax.set_ylabel("F1 Score")
ax.set_title("Neural Network F1 score vs Layer Szie ")
ax.plot(layers, training_scores, marker='o', label="training",
        drawstyle="steps-post")
ax.plot(layers , prediction_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Neutral Network Model Scores.png")


training_time=[]
predict_time=[]
training_scores=[]
prediction_scores=[]
ns=[50,100,200,300,400,500,800,1000]
for n in ns:
    mlp_model=GradientBoostingClassifier(n_estimators=n)
    start=time.time()
    mlp_model.fit(X_train,y_train)
    end=time.time()
    training_scores.append(f1_score(y_train,mlp_model.predict(X_train),average='macro'))
    mlp_model.predict(X_test)
    end2=time.time()
    prediction_scores.append(f1_score(y_test,mlp_model.predict(X_test),average='macro'))

    training_time.append(end-start)
    predict_time.append(end2-end)

fig, ax = plt.subplots()
ax.set_xlabel("Boosting Sizes")
ax.set_ylabel("Time Spent")
ax.set_title("BoostingTraining Time ")
ax.plot(ns, training_time, marker='o', label="training_time",
        drawstyle="steps-post")
ax.plot(ns , predict_time, marker='o', label="prediction_time",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Boosting Model Performance.png")

fig, ax = plt.subplots()
ax.set_xlabel("Boosting  Sizes ")
ax.set_ylabel("F1 Score")
ax.set_title("Boosting F1 score vs Layer Szie ")
ax.plot(ns, training_scores, marker='o', label="training",
        drawstyle="steps-post")
ax.plot(ns , prediction_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig("Boosting Model Scores.png")