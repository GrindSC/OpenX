## 1.Load the Covertype Data Set

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Neural networks
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import Accuracy,Precision,Recall
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# Parameters hypertuning
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Create list of data attributes
attributes='''
Name                                     Data Type    Measurement                       Description
Elevation                               quantitative    meters                       Elevation in meters
Aspect                                  quantitative    azimuth                      Aspect in degrees azimuth
Slope                                   quantitative    degrees                      Slope in degrees
Horizontal_Distance_To_Hydrology        quantitative    meters                       Horz Dist to nearest surface water features
Vertical_Distance_To_Hydrology          quantitative    meters                       Vert Dist to nearest surface water features
Horizontal_Distance_To_Roadways         quantitative    meters                       Horz Dist to nearest roadway
Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice
Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice
Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice
Horizontal_Distance_To_Fire_Points      quantitative    meters                       Horz Dist to nearest wildfire ignition points
Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation
Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation
Cover_Type (7 types)                    integer         1 to 7                       Forest Cover Type designation
'''
attributes=[line.split(' ')[0] for line in attributes.split('\n')[2:-1]]
attributes[-3:-1]=[attributes[-3]+f'_{i}' for i in range(4)]+[attributes[-2]+f'_{i}' for i in range(40)]

# Read data
df = pd.read_csv('covtype.data', header=None)

# Plot correlation between features
plt.figure(figsize=(7,5))
sns.heatmap(df.corr())

# Rename columns
df.columns=attributes

# Plot class distribution
sns.displot(df.Cover_Type)

# Divide data on test/train sets and scale
X = df.drop(columns=['Cover_Type'])
y = df['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.33, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

## 2.Implement a very simple heuristic that will classify the data
# Since it doesn't have to be accurate I will implement Nearest Mean algorithm where I will calculate Euclidean distance between sample and mean class member
# It's similar to Nearest Neighour algorithm but since there is a lot of data it will spead up predictions significantly
    
class NearestMean:
    def __init__(self):
        self.means = None

    def fit(self, X, y):
        y_classes = np.unique(y)
        self.means = [np.mean(X[y == target_class], axis=0) for target_class in y_classes]

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for idx,sample in enumerate(X):
            euclidean_distances = [np.sqrt(np.sum(np.square(sample - mean))) for mean in self.means]
            y_pred[idx]=np.argmin(euclidean_distances)
        return (np.array(y_pred) + 1).astype(int)

def train_and_display(model,type):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm=confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
    print(f'Metrics for {type}:\n{classification_report(y_test, y_pred)}')
    return model

nm_model=train_and_display(NearestMean(),'NearestMean')

## 3.Use Scikit-learn library to train two simple Machine Learning models

# Create class weights which can help with unbalanced data class distribution
unique, counts = np.unique(y_train, return_counts=True)
highest_class_num=max(counts)
class_weights = {key+1: highest_class_num/value for key, value in enumerate(counts)}

rf_model=train_and_display(RandomForestClassifier(n_estimators=100, random_state=42,class_weight=class_weights),'RandomForestClassifier')
lr_model=train_and_display(LogisticRegression(solver='saga',penalty='l1',random_state=42,class_weight=class_weights,max_iter=100),'LogisticRegression')

## 4.Use TensorFlow library to train a neural network that will classify the data

# Transform data for training
y_train = y_train - 1
y_test = y_test - 1
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Define f1 metric
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# inputs=Input(54,)
# x = Dense(256, activation='relu')(inputs)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# outputs = Dense(7, activation='softmax')(x)
# model = Model(inputs=inputs, outputs=outputs)
# optimizer=tf.keras.optimizers.Adam()
# metrics=[
#     'accuracy',
#     f1_metric
# ]
# model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=metrics)
# model.fit(X_train,y_train_onehot, epochs=15,validation_data=(X_test, y_test_onehot))

## 4.1.Create a function that will find a good set of hyperparameters for the NN
# Hyperparameter tuning is long process, so I did initial training to see which parameters migh be worth to tune
# Training process was stable. There was no overfitting and model stoped improving under 20 epochs

def create_model(num_layers, num_neurons):
    neurons=num_neurons
    inputs=Input(54,)
    x=inputs
    for layer in range(num_layers):
      x = Dense(neurons, activation='relu')(x)
      x = BatchNormalization()(x)
      neurons/=2
    outputs = Dense(7, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer=tf.keras.optimizers.Adam()
    metrics=[
        'accuracy',
        f1_metric,
    ]
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=metrics)
    return model

def find_best_hyperparameters(X_train, y_train):
    # Tune number of layers and number of neurons
    num_layers = [1,2,3]
    num_neurons = [256,512,768]

    # Search best parameters
    model = KerasClassifier(build_fn=create_model, epochs=20)
    param_dist = dict(num_layers=num_layers, num_neurons=num_neurons)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, verbose=2, cv=3, n_iter=2)
    random_search.fit(X_train, y_train)
    return random_search.best_params_

best_params=find_best_hyperparameters(X_train, y_train_onehot)

# Create model with best parameters
nn_model=create_model(**best_params)
history=nn_model.fit(X_train,y_train_onehot, epochs=15,validation_data=(X_test, y_test_onehot))

## 4.2.Plot training curves for the best hyperparameters
def plot_training_curves(history):
    fig, axs = plt.subplots(1, 3, figsize=(16, 3),gridspec_kw={"wspace": 0.3})
    # Plot train/test loss
    axs[0].plot(history.history['loss'], label='Training loss')
    axs[0].plot(history.history['val_loss'], label='Validation loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot train/test accuracy
    axs[1].plot(history.history['accuracy'], label='Training accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Plot train/test f1-metric
    axs[2].plot(history.history['f1_metric'], label='Training f1-metric')
    axs[2].plot(history.history['val_f1_metric'], label='Validation f1-metric')
    axs[2].set_title('Training and Validation F1-Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('F1-Score')
    axs[2].legend()

    plt.show()

plot_training_curves(history)

## 5.Evaluate your neural network and other models
y_train = y_train + 1
y_test = y_test + 1

def create_plots():
    models = {'NearestMean': nm_model, 'RandomForestClassifier': rf_model,
              'LogisticRegression': lr_model, 'NeuralNetwork': nn_model}

    # make predictions on the test data
    y_preds = {}
    for name, model in models.items():
        if name == 'NeuralNetwork':
            y_preds[name] = np.argmax(nn_model.predict(X_test,verbose=0),axis=1) + 1
        else:
            y_preds[name] = model.predict(X_test)

    # create a confusion matrix plot for each model
    fig, axs = plt.subplots(2, 2, figsize=(10, 10),gridspec_kw={"hspace": 0.3})
    for i, (name, model) in enumerate(models.items()):
        row = i // 2
        col = i % 2
        cm = confusion_matrix(y_test, y_preds[name])
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=axs[row][col])
        disp.im_.colorbar.remove()
        axs[row][col].set_xlabel('Predicted label')
        axs[row][col].set_ylabel('True label')
        axs[row][col].set_title(name)

    # create a classification report plot for each model
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10),gridspec_kw={"wspace": 0.5})
    for i, (name, model) in enumerate(models.items()):
        row = i // 2
        col = i % 2
        clf_report = classification_report(y_test, y_preds[name],
                                      target_names=[str(i) for i in range(1,8)],
                                      output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, ax=axs2[row][col],vmin=0, vmax=1, cbar=False)
        axs2[row][col].set_title(name)
        fig2.subplots_adjust(right=0.8)
        cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
        sm = plt.cm.ScalarMappable(cmap=sns.color_palette("rocket", as_cmap=True))
        sm.set_array([])
        fig2.colorbar(sm, cax=cbar_ax)

create_plots()

# Save model for the REST API
pickle.dump(nm_model, open('nm_model.pkl', 'wb'))
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
pickle.dump(lr_model, open('lr_model.pkl', 'wb'))
nn_model.save('nn_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)