from sklearn import linear_model,neighbors,svm,preprocessing,tree,ensemble,tree,metrics
import pandas
import numpy
import xgboost as xgb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import os

class trainning_netflix:
    def __init__(self,db,pct,col_pred,nb_couches,nb_var_select,nb_var_couche,cv,nb_split,prop_db,model_type):
        self.db = db
        self.pct = pct
        self.x_train = pandas.DataFrame()
        self.x_test = pandas.DataFrame()
        self.y_train = pandas.DataFrame()
        self.y_test = pandas.DataFrame()
        self.col_pred = col_pred
        self.nb_couches = nb_couches
        self.nb_var_select = nb_var_select
        self.nb_var_couche = nb_var_couche
        self.cv = cv
        self.mes = pandas.DataFrame()
        self.nb_split = nb_split
        self.prop_db = prop_db
        self.model_type = model_type
        self.mes['prop'] = [self.prop_db]
        self.mes['model'] = self.model_type
        self.mes['all_1'] = 0
        self.mes['all_0'] = 0
        for d in range(len(self.col_pred)):
            self.mes[str(d) + '_1'] = 0
            self.mes[str(d) + '_0'] = 0
            self.mes[str(d) + '_auc'] = 0

    def split_train_test(self):
        rand = numpy.random.choice(range(len(self.db)),size = int((1 - self.pct) * len(self.db)),replace = False)
        self.x_train = self.db.iloc[rand]
        self.y_train = self.x_train[self.col_pred]
        self.x_train = self.x_train.drop(self.col_pred,axis = 1)
        self.x_test = self.db.drop(rand,axis = 0)
        self.y_test = numpy.array(self.x_test[self.col_pred])
        self.x_test = self.x_test.drop(self.col_pred,axis = 1)

    def th_evolution(self):
        for i in range(self.nb_couches):
            n_x_train = numpy.array([])
            n_x_test = numpy.array([])
            col_x = list(self.x_train.columns)
            for j in range(self.nb_var_couche[i]):
                col_rand = numpy.random.choice(col_x,size = min(len(col_x),self.nb_var_select),replace = False)
                for c in col_rand:
                    col_x.remove(c)
                model = linear_model.LinearRegression()
                # model = linear_model.LogisticRegression(penalty='none',solver='newton-cg')
                x_an_train = self.x_train[col_rand]
                x_an_test = self.x_test[col_rand]
                model.fit(x_an_train,self.y_train)
                pred_train = model.predict(x_an_train)[:,0]
                pred_test = model.predict(x_an_test)[:,0]
                if n_x_train.shape[0] == 0:
                    n_x_train = pred_train
                    n_x_test = pred_test
                else:
                    n_x_train = numpy.c_[n_x_train,pred_train]
                    n_x_test = numpy.c_[n_x_test,pred_test]
            self.x_test = pandas.DataFrame(n_x_test)
            self.x_train = pandas.DataFrame(n_x_train)
        model = linear_model.LogisticRegression(penalty='none',solver='newton-cg')
        model.fit(self.x_train,self.y_train)
        return model.predict_proba(self.x_test)[:,1]

    def mesure_1(self):
        for i in range(self.cv):
            self.split_train_test()
            k_y = 1
            y_pred = None
            for col_y in self.y_train:
                cor_xy = []
                for c in self.x_train:
                    if numpy.var(self.x_train[c]) > 0:
                        self.x_train[c] = (numpy.array(self.x_train) - numpy.mean(self.x_train)) / numpy.var(self.x_train)
                model = linear_model.LinearRegression()
                model.fit(self.x_train,self.y_train[col_y])
                cor_xy = numpy.abs(model.coef_)

                # for c in self.x_train:
                #     cor_xy.append(abs(self.y_train[col_y].corr(self.x_train[c])))
                # cor_xy = numpy.array(cor_xy)
                # sel = cor_xy.astype(str) == 'nan'
                # cor_xy[sel] = 0

                cor_xy = pandas.DataFrame(cor_xy)
                cor_xy.columns = ['corr']
                cor_xy = cor_xy.sort_values(['corr'],ascending = False)
                k = 1
                col_x = self.x_train.columns
                
                while k <= self.nb_split:
                    x_an_train = self.x_train[col_x[cor_xy.index[0:int(k / self.nb_split * len(col_x))]]]
                    x_an_test = self.x_test[col_x[cor_xy.index[0:int(k / self.nb_split * len(col_x))]]]
                    
                    if self.model_type == 'logistic':
                        model = linear_model.LogisticRegression(penalty='none',solver='newton-cg')
                    elif self.model_type == 'tree':
                        model = tree.DecisionTreeClassifier()
                    elif self.model_type == 'rf':
                        model = ensemble.RandomForestClassifier()
                    elif self.model_type == 'svm':
                        model = svm.SVC(probability = True)
                    elif self.model_type == 'xgboost':
                        model = xgb.XGBClassifier(objective="binary:logistic")
                    
                    if  self.model_type != 'deeplearning':
                        model.fit(x_an_train,self.y_train[col_y])
                        y_pred = model.predict_proba(x_an_test)[:,1]

                    if self.model_type == 'deeplearning':
                        model = Sequential()
                        model.add(Dense(128, activation='relu'))
                        # model.add(Dense(128, activation='softplus'))
                        # model.add(Dense(128, activation='tanh'))
                        # model.add(Dropout(0.5))
                        model.add(Dense(1, activation='sigmoid'))
                        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
                        model.fit(numpy.array(x_an_train), numpy.array(self.y_train[col_y]),batch_size=32, nb_epoch=10, verbose=0)
                        y_pred = model.predict_proba(x_an_test)[:,0]

                    fpr, tpr, thresholds = metrics.roc_curve(self.y_test[:,0], y_pred, pos_label=1)
                    auc_res = metrics.auc(fpr, tpr)
                    sel_11 = (y_pred >= 0.5) *  (self.y_test[:,0] == 1)
                    sel_10 = (y_pred >= 0.5) *  (self.y_test[:,0] == 0)
                    sel_01 = (y_pred < 0.5) * (self.y_test[:,0] == 1)
                    sel_00 = (y_pred < 0.5) * (self.y_test[:,0] == 0)
                    self.mes['11'].iloc[k-1] += sum(sel_11)
                    self.mes['10'].iloc[k-1] += sum(sel_10)
                    self.mes['01'].iloc[k-1] += sum(sel_01)
                    self.mes['00'].iloc[k-1] += sum(sel_00)
                    self.mes['auc'].iloc[k-1] += auc_res
                    print(str(k_y) + ' - ' + str(k) + ' - ' + str(i))
                    k += 1
                k_y += 1
        self.mes.to_csv('C:/netflix/model_outecomes_imdb.csv',sep = ';',index = False)

    def mesure(self):
        for i in range(self.cv):
            self.split_train_test()
            k_y = 1
            y_pred = None
            col_y_i = 0
            for col_y in self.y_train:
                cor_xy = []
                # for c in self.x_train:
                #     if numpy.var(self.x_train[c]) > 0:
                #         self.x_train[c] = (numpy.array(self.x_train[c]) - numpy.mean(self.x_train[c])) / numpy.var(self.x_train[c])
                # model = linear_model.LinearRegression()
                # model.fit(self.x_train,self.y_train[col_y])
                # cor_xy = numpy.abs(model.coef_)

                for c in self.x_train:
                    cor_xy.append(abs(self.y_train[col_y].corr(self.x_train[c])))
                cor_xy = numpy.array(cor_xy)
                sel = cor_xy.astype(str) == 'nan'
                cor_xy[sel] = 0

                cor_xy = pandas.DataFrame(cor_xy)
                cor_xy.columns = ['corr']
                cor_xy = cor_xy.sort_values(['corr'],ascending = False)
                k = 1
                col_x = self.x_train.columns
                
                x_an_train = self.x_train[col_x[cor_xy.index[0:int(self.prop_db * len(col_x))]]]
                x_an_test = self.x_test[col_x[cor_xy.index[0:int(self.prop_db * len(col_x))]]]
                
                if self.model_type == 'logistic':
                    model = linear_model.LogisticRegression(penalty='none',solver='newton-cg')
                elif self.model_type == 'tree':
                    model = tree.DecisionTreeClassifier()
                elif self.model_type == 'rf':
                    model = ensemble.RandomForestClassifier()
                elif self.model_type == 'svm':
                    model = svm.SVC(probability = True)
                elif self.model_type == 'xgboost':
                    model = xgb.XGBClassifier(objective="binary:logistic")
                
                if  self.model_type != 'deeplearning':
                    model.fit(x_an_train,self.y_train[col_y])
                    try:
                        y_pred = numpy.c_[y_pred,model.predict_proba(x_an_test)[:,1]]
                    except:
                        y_pred = model.predict_proba(x_an_test)[:,1]
                
                if self.model_type == 'deeplearning':
                    model = Sequential()
                    model.add(Dense(128, activation='relu'))
                    # model.add(Dense(128, activation='softplus'))
                    # model.add(Dense(128, activation='tanh'))
                    # model.add(Dropout(0.5))
                    model.add(Dense(1, activation='sigmoid'))
                    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
                    model.fit(numpy.array(x_an_train), numpy.array(self.y_train[col_y]),batch_size=32, nb_epoch=10, verbose=0)

                    try:
                        y_pred = numpy.c_[y_pred,model.predict_proba(x_an_test)[:,0]]
                    except:
                        y_pred = model.predict_proba(x_an_test)[:,0]
                try:
                    fpr, tpr, thresholds = metrics.roc_curve(self.y_test[:,col_y_i], y_pred, pos_label=1)
                    auc_res = metrics.auc(fpr, tpr)
                except:
                    fpr, tpr, thresholds = metrics.roc_curve(self.y_test[:,col_y_i], y_pred[:,y_pred.shape[1] - 1], pos_label=1)
                    auc_res = metrics.auc(fpr, tpr)

                self.mes[str(col_y_i) + '_auc'].iloc[0] += auc_res

                col_y_i += 1

                print(str(k_y) + ' - ' + str(self.prop_db) + ' - ' + str(i))
                k_y += 1

            y_dec = numpy.apply_along_axis(numpy.argmax,1,y_pred)
            kyd = 0
            for d in y_dec:
                self.mes['all_1'].iloc[0] += self.y_test[kyd,d]
                self.mes['all_0'].iloc[0] += 1 - self.y_test[kyd,d]
                self.mes[str(d) + '_1'] += self.y_test[kyd,d]
                self.mes[str(d) + '_0'].iloc[0] += 1 - self.y_test[kyd,d]

                kyd += 1

        self.mes.to_csv('C:/fake_news/model_outecomes_' + str(self.model_type) + '_' + str(self.prop_db) + '.csv',sep = ';',index = False)

model_type = ['deeplearning','rf','logistic','xgboost']
model_type = ['rf']
db = pandas.read_csv('C:/fake_news/doc/bdd_text.csv',engine = 'python',sep = ';')
sel_nan = db['note_fb'].apply(lambda x: str(x).lower() != 'nan')
db = db[sel_nan]
db.index = range(len(db))

col_pred = []
nb_split = 2
seuil = 0
for q in range(nb_split):
    # sel_q = numpy.array(db['note_fb'] <= numpy.quantile(numpy.array(db['note_fb']),(q+1) / nb_split)) * numpy.array(db['note_fb'] > numpy.quantile(numpy.array(db['note_fb']),q / nb_split))
    sel_q = numpy.array(db['note_fb'] > seuil)
    if q == 0:
        sel_q = (sel_q == False)
    db['note_' + str(q)] = 0
    db['note_' + str(q)][sel_q]= 1
    col_pred.append('note_' + str(q))

db = db.drop('note_fb',axis = 1)

nb_var_couche = [0]
nb_var_select = 10
nb_couches = len(nb_var_couche)
nb_split_cor = 10
for prop in [1]:
    for m in model_type:
        tn = trainning_netflix(db = db,
                                pct = 0.1,
                                col_pred = col_pred,
                                nb_couches = nb_couches,
                                nb_var_select = nb_var_select,
                                nb_var_couche = nb_var_couche,
                                cv = 10,
                                nb_split = 10,
                                prop_db = (prop + 1) / nb_split_cor,
                                model_type = m)
        tn.mesure()

mat = pandas.DataFrame()
for prop in range(100):
    for m in model_type:
        try:
            filename = 'C:/fake_news/model_outecomes_' + str(m) + '_' + str((prop + 1) / 100) + '.csv'
            temp = pandas.read_csv(filename,engine = 'python',sep = ';')
            temp = temp.iloc[0]
            mat = mat.append(temp)
            os.remove(filename)
        except:
            pass
mat.to_csv('C:/fake_news/model_outecomes_' + str(q+1) + '.csv',index = False,sep = ';')
