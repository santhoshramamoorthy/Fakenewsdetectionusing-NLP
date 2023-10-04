# %% [markdown]
# If you're going to make a copy can you please upvote this notebook (im trying to make progress)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:10.6943Z","iopub.execute_input":"2023-07-23T12:20:10.69475Z","iopub.status.idle":"2023-07-23T12:20:19.300341Z","shell.execute_reply.started":"2023-07-23T12:20:10.694698Z","shell.execute_reply":"2023-07-23T12:20:19.299275Z"}}
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

import itertools
import seaborn as sns
from sklearn import metrics 

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:19.302575Z","iopub.execute_input":"2023-07-23T12:20:19.303436Z","iopub.status.idle":"2023-07-23T12:20:21.884236Z","shell.execute_reply.started":"2023-07-23T12:20:19.303399Z","shell.execute_reply":"2023-07-23T12:20:21.880503Z"}}
df_true1 = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
df_fake1 = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:21.885831Z","iopub.execute_input":"2023-07-23T12:20:21.886406Z","iopub.status.idle":"2023-07-23T12:20:21.901396Z","shell.execute_reply.started":"2023-07-23T12:20:21.886371Z","shell.execute_reply":"2023-07-23T12:20:21.900219Z"}}
print(df_true1["text"][0])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:21.904552Z","iopub.execute_input":"2023-07-23T12:20:21.905401Z","iopub.status.idle":"2023-07-23T12:20:21.920548Z","shell.execute_reply.started":"2023-07-23T12:20:21.905368Z","shell.execute_reply":"2023-07-23T12:20:21.919173Z"}}
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:21.922644Z","iopub.execute_input":"2023-07-23T12:20:21.923178Z","iopub.status.idle":"2023-07-23T12:20:52.682672Z","shell.execute_reply.started":"2023-07-23T12:20:21.923146Z","shell.execute_reply":"2023-07-23T12:20:52.681705Z"}}
#texts = df_true1["title"].to_list()

fake_articlesdf = pd.read_csv("/kaggle/input/chatgpt-fake-articles/chatgpt_fake_articles (1).csv", encoding='unicode_escape')

fake_articles = fake_articlesdf["text"].to_list()

fake_articlesdf2 = pd.read_csv("/kaggle/input/chatgpt-fake-articles/chatgpt_fake_articles.csv", encoding='unicode_escape')

fake_articles2 = fake_articlesdf2["text"].to_list()



true_titles = df_true1["text"].to_list()

fake_titles = df_fake1["text"].to_list()

texts = true_titles + fake_titles + fake_articles

#add a new dataset to this one to test (so the tokenizer has seen the words before)
textswithfakearticles = true_titles + fake_titles + fake_articles + fake_articles2

# Create labels
true_labels = [1] * len(true_titles)
fake_labels = [0] * len(fake_titles)
fake_article_labels = [0] * len(fake_articles)

labels = true_labels + fake_labels + fake_article_labels




#print(texts)

#texts = [text]

#print(texts)

tokenizer.fit_on_texts(textswithfakearticles)

tokenized = tokenizer.texts_to_sequences(texts)



print(max(tokenizer.word_index.values()))

vocab_size = max(tokenizer.word_index.values())+1

print(vocab_size)


#print(tokenized)

print(len(tokenized))
print(len(labels))
#print(labels)



# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:52.684072Z","iopub.execute_input":"2023-07-23T12:20:52.685037Z","iopub.status.idle":"2023-07-23T12:20:53.732313Z","shell.execute_reply.started":"2023-07-23T12:20:52.685002Z","shell.execute_reply":"2023-07-23T12:20:53.731408Z"}}
word_counts = tokenizer.word_counts
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
most_common_words = sorted_word_counts[:100]
words, counts = zip(*most_common_words)
plt.figure(figsize=(20,10))
plt.bar(words, counts)
plt.xticks(rotation='vertical')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T13:12:11.752241Z","iopub.execute_input":"2023-07-23T13:12:11.752616Z","iopub.status.idle":"2023-07-23T13:12:25.370748Z","shell.execute_reply.started":"2023-07-23T13:12:11.752585Z","shell.execute_reply":"2023-07-23T13:12:25.369726Z"}}
lengths = [len(item) for item in tokenized]

max_length = max(lengths)
min_length = min(lengths)

max_length_index = lengths.index(max_length)
min_length_index = lengths.index(min_length)

true_tokenized = tokenizer.texts_to_sequences(true_titles)

fake_tokenized = tokenizer.texts_to_sequences(fake_titles)

true_lengths = [len(item) for item in true_tokenized]

fake_lengths = [len(item) for item in fake_tokenized]

#what i find incredibly ironic is that if you actually run the two comments below
#specifically the first one is that the article you get is about fake news

#print(f"Longest element is '{texts[max_length_index]}' with length {max_length} at index {max_length_index}")
#print(f"Shortest element is '{texts[min_length_index]}' with length {min_length} at index {min_length_index}")

print("Distribution of article lengths")

#sns.kdeplot(lengths, label='Overall')
sns.kdeplot(true_lengths, label='Real')
sns.kdeplot(fake_lengths, label='Fake')

plt.legend()
plt.show()

# %% [markdown]
# It's quite surprising that Fake articles don't have that spike around low lengths that Real ones do.

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:54.375739Z","iopub.execute_input":"2023-07-23T12:20:54.376391Z","iopub.status.idle":"2023-07-23T12:20:56.927461Z","shell.execute_reply.started":"2023-07-23T12:20:54.376355Z","shell.execute_reply":"2023-07-23T12:20:56.92642Z"}}
from tensorflow.keras.preprocessing.sequence import pad_sequences


padded_sequences = pad_sequences(tokenized, padding='post', maxlen=max_length)

print(padded_sequences)

lengths = [len(item) for item in padded_sequences]

max_length = max(lengths)
min_length = min(lengths)

max_length_index = lengths.index(max_length)
min_length_index = lengths.index(min_length)

#print(f"Longest element is '{texts[max_length_index]}' with length {max_length} at index {max_length_index}")
#print(f"Shortest element is '{texts[min_length_index]}' with length {min_length} at index {min_length_index}")


# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:20:56.931989Z","iopub.execute_input":"2023-07-23T12:20:56.934453Z","iopub.status.idle":"2023-07-23T12:21:03.951522Z","shell.execute_reply.started":"2023-07-23T12:20:56.934414Z","shell.execute_reply":"2023-07-23T12:21:03.950559Z"}}
embedding_length = 44
#embedding_length = 10
#embedding_length = 1

from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * (10**(-epoch/2))
callbacks = [LearningRateScheduler(scheduler, verbose=1)]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_length, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences = False)))
#model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)))
#model.add(tf.keras.layers.Dropout(0.15)) #speculative
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation = "relu"))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(2, activation = "softmax"))
input_array = np.random.randint(1000, size=(32, max_length))
model.compile(tf.keras.optimizers.AdamW(weight_decay=0.001), 'binary_focal_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
output_array = model.predict(input_array)
print(output_array.shape)
model.summary()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:21:03.957347Z","iopub.execute_input":"2023-07-23T12:21:03.958028Z","iopub.status.idle":"2023-07-23T12:21:03.964148Z","shell.execute_reply.started":"2023-07-23T12:21:03.957992Z","shell.execute_reply":"2023-07-23T12:21:03.963151Z"}}
print(padded_sequences[20])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:21:03.965477Z","iopub.execute_input":"2023-07-23T12:21:03.966065Z","iopub.status.idle":"2023-07-23T12:35:31.534162Z","shell.execute_reply.started":"2023-07-23T12:21:03.966028Z","shell.execute_reply":"2023-07-23T12:35:31.532171Z"}}
labels = np.array(labels)

one_hot_labels = to_categorical(labels)

print(one_hot_labels[0])

sequences_train, sequences_test, labels_train, labels_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=43)


#model.fit(sequences_train, labels_train, epochs = 3, batch_size = 256, validation_data=(sequences_test, labels_test), callbacks=callbacks)
model.fit(sequences_train, labels_train, epochs = 4, batch_size = 64, validation_data=(sequences_test, labels_test), callbacks=callbacks)
model.evaluate(sequences_test, labels_test)




# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T12:35:47.786053Z","iopub.execute_input":"2023-07-23T12:35:47.78649Z","iopub.status.idle":"2023-07-23T12:40:01.536772Z","shell.execute_reply.started":"2023-07-23T12:35:47.786454Z","shell.execute_reply":"2023-07-23T12:40:01.535776Z"}}
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Dark2):
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks=np.arange(len(classes))
  plt.xticks(tick_marks,classes,rotation=45)
  plt.yticks(tick_marks,classes)
  if normalize:
    cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    print("normalize confusion matrix")
  else:
    print('confusion matrix,without normalization')

  thresh = cm.max() / 2
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="yellow" if cm[i, j] > thresh else "red")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label') 

pred_lr=model.predict(sequences_test)
score = model.evaluate(sequences_test, labels_test)
pred_lr = (pred_lr > 0.5).astype(int)
labels_test_single = np.argmax(labels_test, axis=1)
pred_lr_single = np.argmax(pred_lr, axis=1)
print(f'Accuracy: {round(score[1]*100,2)}%')
cm = metrics.confusion_matrix(labels_test_single,pred_lr_single)
plot_confusion_matrix(cm, classes=['FAKE NEWS', 'REAL NEWS'])


# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T13:04:25.376898Z","iopub.execute_input":"2023-07-23T13:04:25.37729Z","iopub.status.idle":"2023-07-23T13:04:25.418425Z","shell.execute_reply.started":"2023-07-23T13:04:25.377259Z","shell.execute_reply":"2023-07-23T13:04:25.41728Z"}}
misclassified_indices = np.where(pred_lr_single != labels_test_single)[0]
#print(misclassified_indices)

incorrect_labels = []
for i in misclassified_indices:
    incorrect_labels.append(pred_lr_single[i])

misclassified_sequences = []
for i in misclassified_indices:
    misclassified_sequences.append(sequences_test[i])
misclassified_texts = tokenizer.sequences_to_texts(misclassified_sequences)

for i in range(len(misclassified_texts)):
    if incorrect_labels[i] == 1:
        print("Predicted Real\n")
    if incorrect_labels[i] == 0:
        print("Predicted Fake\n")
    print(misclassified_texts[i])
    print()

    


# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T13:04:30.693768Z","iopub.execute_input":"2023-07-23T13:04:30.694181Z","iopub.status.idle":"2023-07-23T13:04:31.077248Z","shell.execute_reply.started":"2023-07-23T13:04:30.694144Z","shell.execute_reply":"2023-07-23T13:04:31.076265Z"}}
#you can use this cell to look into different inputs and their corresponding predictions
#the two marked positions below indicate the demarkation between real and fake news
#the label [0, 1] represents real news

position = 21416

position = 21417

input_array=padded_sequences[position]
print(texts[position])
print(input_array)

input_array = np.expand_dims(input_array, 0)
print(input_array)

output_array = model.predict(input_array)
print(output_array)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T13:27:43.920664Z","iopub.execute_input":"2023-07-23T13:27:43.921647Z","iopub.status.idle":"2023-07-23T13:27:43.930006Z","shell.execute_reply.started":"2023-07-23T13:27:43.921602Z","shell.execute_reply":"2023-07-23T13:27:43.928936Z"}}
hard_test = tokenizer.texts_to_sequences(fake_articles2)

padded_fake_articles = pad_sequences(hard_test, padding='post', maxlen=8375)



# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T13:27:45.477877Z","iopub.execute_input":"2023-07-23T13:27:45.478246Z","iopub.status.idle":"2023-07-23T13:27:46.533596Z","shell.execute_reply.started":"2023-07-23T13:27:45.478212Z","shell.execute_reply":"2023-07-23T13:27:46.532693Z"}}
#this is a test on fake articles generated by chatGPT

#for position in range(len(padded_fake_articles)):
for position in range(2):
    input_array=padded_fake_articles[position]
    print(fake_articles[position])
    #print(input_array)

    input_array = np.expand_dims(input_array, 0)
    #print(input_array)

    output_array = model.predict(input_array)
    print(output_array)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-23T13:27:48.809951Z","iopub.execute_input":"2023-07-23T13:27:48.81031Z","iopub.status.idle":"2023-07-23T13:27:50.140947Z","shell.execute_reply.started":"2023-07-23T13:27:48.810281Z","shell.execute_reply":"2023-07-23T13:27:50.139988Z"}}
hard_test_labels = [0] * len(padded_fake_articles)

hard_test_labels = [1] + hard_test_labels

hard_test_labels = np.array(hard_test_labels)

one_hot_hard_test_labels = to_categorical(hard_test_labels)

one_hot_hard_test_labels = np.delete(one_hot_hard_test_labels, 0, axis=0)


#print(one_hot_hard_test_labels)

model.evaluate(padded_fake_articles, one_hot_hard_test_labels)



pred_lr2=model.predict(padded_fake_articles)
score2 = model.evaluate(padded_fake_articles, one_hot_hard_test_labels)
pred_lr2 = (pred_lr2 > 0.5).astype(int)
labels_test_single2 = np.argmax(one_hot_hard_test_labels, axis=1)
pred_lr_single2 = np.argmax(pred_lr2, axis=1)
print(f'Accuracy: {round(score2[1]*100,2)}%')
cm = metrics.confusion_matrix(labels_test_single2,pred_lr_single2)
plot_confusion_matrix(cm, classes=['FAKE NEWS', 'REAL NEWS'])
