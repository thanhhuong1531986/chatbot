# CNN for the IMDB problem
import numpy
import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping



import math
import random
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

responses=[]
name_classification=[]
def concat_lines(lines_str):
    lines = lines_str.split('\n')
    str_lines = lines[0]
    for i in range(1,len(lines)):
        if len(lines[i])>0:
            str_lines = str_lines + ' ; ' + lines[i]   
    return str_lines

LIST_SHEET_NAME = ['CTDT_HTTT_2018','Ho_tro_tuyen_sinh','van_ban','Giang_vien']

def load_data():
    X_train_data=[]
    Y_train_data=[]
    class_data=-1
    for s in range(4):
        df = pd.read_excel('data\\build_data.xlsx', sheet_name=LIST_SHEET_NAME[s])
        for i in df.index:
            if math.isnan(df['stt'][i]):
                X_train_data.append(concat_lines(str(df['sample'][i])))
                Y_train_data.append('c_'+str(class_data))
            else:
                responses.append(concat_lines(str(df['answer'][i])))
                name_classification.append(concat_lines(str(df['entity_faq_keyword'][i])))
                class_data=class_data+1
                X_train_data.append(concat_lines(str(df['question'][i])))
                Y_train_data.append('c_'+str(class_data))
                X_train_data.append(concat_lines(str(df['sample'][i])))
                Y_train_data.append('c_'+str(class_data))
    return X_train_data, Y_train_data

def get_list_test(dtset):
    sz = int(len(dtset) * 0.15)
    return random.sample([i for i in range(len(dtset))],sz)

def creat_data_test(X, Y):
    list_X =[]
    list_Y =[]
    list_dt =[]
    list_dt = get_list_test(X)
    for i in range(len(X)):
        if i in list_dt:
            list_X.append(X[i])
            list_Y.append(Y[i])
    return list_X, list_Y

X_train, Y_train = load_data()
X_test, Y_test = creat_data_test(X_train, Y_train)

df = pd.DataFrame(X_train,columns=['Questions'])
df['Class'] = Y_train
list_indexs_class = df.Class.value_counts()
#print(list_indexs_class.index[0])
num_of_categories = len(list_indexs_class)
shuffled = df.reindex(numpy.random.permutation(df.index))

list_classes = []
for i in range(len(df.Class.value_counts())):
    lc = shuffled[shuffled['Class'] == list_indexs_class.index[i]][:num_of_categories]
    list_classes.append(lc)
concated = pd.concat(list_classes, ignore_index=True)
#Shuffle the dataset
concated = concated.reindex(numpy.random.permutation(concated.index))
concated['LABEL'] = 0
for i in range(len(df.Class.value_counts())):
    l_i_c = list_indexs_class.index[i]
    concated.loc[concated['Class'] == l_i_c, 'LABEL'] = int(l_i_c[2:])
print(concated['LABEL'][:10])
labels = to_categorical(concated['LABEL'], num_classes=len(list_indexs_class))
print(labels[:10])
if 'Class' in concated.keys():
    concated.drop(['Class'], axis=1)

n_most_common_words = 8000
max_len = 600
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(concated['Questions'].values)
sequences = tokenizer.texts_to_sequences(concated['Questions'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)


X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.15, random_state=42)

epochs = 10
emb_dim = 128
batch_size = 256
labels[:2]

print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.7))
model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(120, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
#history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test, y_test),callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

#history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

txt = ['Số tín chỉ Kiến thức cơ sở ngành']
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len)
print(padded)
pred = model.predict(padded)
#labels = ['entertainment', 'bussiness', 'science/tech', 'health']
print(pred, responses[numpy.argmax(pred)])
