# python version 3.11.3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from PIL import Image

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from keras.models import Model, load_model
from keras.layers import Dense, Flatten
from keras.metrics import AUC

from keras.applications.vgg19 import VGG19

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import pickle
import random

random.seed(0) # fix the randomness

#----------------------------------------------------------------------------------------------------

def read_data(path):
    """This function retrieves the DataFrame by reading the data from the CSV file."""
    dataframe = pd.read_csv(path)
    return dataframe

#----------------------------------------------------------------------------------------------------

def sorted_images(df):
    """This function facilitates the selection of image_ids specifically from the year 2017 and subsequently removes the '2017' identifier from these image_ids."""
    
    new_df = df[df['image_id'].str.endswith("17.jpg")]
    copied_df = new_df.copy()
    copied_df['image_id'] = copied_df['image_id'].str.replace("2017.jpg", "")
    return copied_df

#----------------------------------------------------------------------------------------------------

def save_train_test_labels(df):
    """This function enables the segregation of the initial 500 images as a test dataset, while the remaining images are saved as a train dataset."""
    
    test_label = df.iloc[:500]
    test_label.to_csv("test_label.csv", index=False)
    train_label = df.iloc[500:]
    train_label.to_csv("train_label.csv", index=False)

    print("Files saved.")

#----------------------------------------------------------------------------------------------------

def dataframes_by_score(df):
    """This function facilitates the partitioning of the training label dataframe based on the confidence scores.
    The resulting dataframes are as follows:
    - the first dataframe comprises images with scores below 1
    - the second dataframe includes images with scores above 0.8, specifically containing oil palm
    - the third dataframe consists of oil palm images with a confidence score of 1
    - the last dataframe comprises images devoid of oil palm with a confidence score of 1."""
    
    score_und_1 = df[df["score"] < 1]
    score_abv_80 = score_und_1[(score_und_1['has_oilpalm']==1)&(score_und_1['score']>=0.8)]
    has_oilpalm = df[(df["score"] == 1)&(df["has_oilpalm"] == 1)]
    no_oilpalm = df[(df['score']==1)&(df['has_oilpalm']==0)]
    return score_und_1, score_abv_80, has_oilpalm, no_oilpalm

#----------------------------------------------------------------------------------------------------

def num_values(df, df_2, df_3, df_4):
    """This function provides the count of rows within the dataframes."""
    
    print(f"There are {df.shape[0]} images in first dataframe.")
    print(f"There are {df_2.shape[0]} images in second dataframe.")
    print(f"There are {df_3.shape[0]} images in third dataframe.")
    print(f"There are {df_4.shape[0]} images in fourth dataframe.")

#----------------------------------------------------------------------------------------------------

def show_images(dataframe):
    """This function aids in visualizing the images corresponding to the separated dataframes."""
    
    random.seed(0)
    
    df_ids = dataframe['image_id'].values
    df_paths = []

    for id in df_ids:
        path = "C:/Users/nene0/Downloads/widsdatathon2019/train_images/" + id + ".jpg"
        df_paths.append(path)
    
    try:
        df_img = random.sample(range(len(df_paths)), 25)

        plt.figure(figsize=(10, 10))
        for i, index in enumerate(df_img, 1):
            img = image.load_img(df_paths[index])
            plt.subplot(5, 5, i)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    except:
        df_img = random.sample(range(len(df_paths)), 16)

        plt.figure(figsize=(8, 8))
        for i, index in enumerate(df_img, 1):
            img = image.load_img(df_paths[index])
            plt.subplot(4, 4, i)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

#----------------------------------------------------------------------------------------------------

def image_to_array(file_paths, target_size):
    """This function utilizes the Keras image module to load an image and subsequently returns the image's pixel values as a NumPy array."""
    
    my_images = []
    for path in file_paths:
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        my_images.append(img_array)
    return np.array(my_images)

#----------------------------------------------------------------------------------------------------

def final_dataframe_ds(oilpalm_df, oilpalm_df_2, no_oilpalm_df):
    """This function returns the downsampled final training dataframe along with a target dataframe."""
    
    no_oilpalm_balanced = no_oilpalm_df.sample(oilpalm_df.shape[0] + oilpalm_df_2.shape[0], random_state=0)
    final_train_labels = pd.concat([oilpalm_df, no_oilpalm_balanced, oilpalm_df_2], ignore_index=True)
    final_train_labels = final_train_labels.sample(frac=1, random_state=0).reset_index(drop=True)

    final_target_labels = final_train_labels[['has_oilpalm']]

    return final_train_labels, final_target_labels

#----------------------------------------------------------------------------------------------------

def image_data_cleaning_ds(df):
    """This function provides the processed final training data in a NumPy format after performing data cleaning and downsampling."""
    
    train_ids = df['image_id'].values

    train_paths =[]

    for id in train_ids:
        path = "C:/Users/nene0/Downloads/widsdatathon2019/train_images/" + id + ".jpg"
        train_paths.append(path)
    
    train = image_to_array(train_paths, (224,224,3))
    train = train/255 # Scale the data.

    return train

#----------------------------------------------------------------------------------------------------

def save_rotated_oilpalm_images(df):
    """This function saves the rotated oil palm images using the PIL library's Image"""
    
    df_ids = df['image_id'].values
    df_paths = []

    for id in df_ids:
        path = "C:/Users/nene0/Downloads/widsdatathon2019/train_images/" + id + ".jpg"
        df_paths.append(path)
    
    for path in df_paths:
        rotate_image = Image.open(path).rotate(90)
        new_file_path = path.replace('.jpg', '_rotated.jpg')
        rotate_image.save(new_file_path)

#----------------------------------------------------------------------------------------------------

def final_dataframe_os(oilpalm_df, no_oilpalm_df):
    """This function generates the finalized dataframes containing train and target labels specifically designed for the oversampling method."""
    
    no_oilpalm_500 = no_oilpalm_df.sample(500, random_state=0) # randomly select 500 images

    rotated_oilpalm = oilpalm_df.copy()
    rotated_oilpalm['image_id'] = rotated_oilpalm['image_id'] + "_rotated"

    total_oilpalm = pd.concat([oilpalm_df, rotated_oilpalm], ignore_index=True).sample(500, random_state=0)

    final_train_labels = pd.concat([no_oilpalm_500, total_oilpalm], ignore_index=True) # concat images totaling 1,000
    final_train_labels = final_train_labels.sample(frac=1, random_state=0).reset_index(drop=True)

    final_target_labels = final_train_labels[['has_oilpalm']]

    return final_train_labels, final_target_labels

#----------------------------------------------------------------------------------------------------

def image_data_cleaning_os(df):
    """This function provides the processed final training data in a NumPy format after performing data cleaning and oversampling."""

    train_ids = df['image_id'].values

    train_paths =[]

    for id in train_ids:
        path = "C:/Users/nene0/Downloads/widsdatathon2019/train_images/" + id + ".jpg"
        train_paths.append(path)

    train = image_to_array(train_paths, (224,224,3))
    train = train/255

    return train

#----------------------------------------------------------------------------------------------------

def build_model():
    """This function provides the final model for recognizing oil palm plantations in images."""

    vgg_19 = VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
    flatten_layer = Flatten()(vgg_19.layers[-1].output)

    dense_layer_1 = Dense(512, activation='relu')(flatten_layer)
    dense_layer_2 = Dense(256, activation='relu')(dense_layer_1)
    dense_layer_3 = Dense(256, activation='relu')(dense_layer_2)
    dense_layer_4 = Dense(128, activation='relu')(dense_layer_3)
    dense_layer_5 = Dense(128, activation='relu')(dense_layer_4)

    output_layer = Dense(1, activation='sigmoid')(dense_layer_5)

    model = Model(inputs=vgg_19.inputs, outputs=output_layer)

    return model

#----------------------------------------------------------------------------------------------------

def train_model(x, y, ann_model, num_batch_size, num_epochs):
  """This function facilitates model training with the specified parameters:
  - Learning rate set to 0.0001
  - Callback configured with a patience value of 5
  - Metrics as AUC"""

  opt = Adam(learning_rate=0.0001)
  callback = EarlyStopping(monitor='val_loss', patience=5)
  ann_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC()])
  history = ann_model.fit(x, y, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.2, callbacks=[callback])
  return history

#----------------------------------------------------------------------------------------------------

def loss_graph(history):
    """This function displays the training and validation loss graphs for the model during its training process."""

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Plot for the train and test(validation) loss")
    plt.legend(['train','test'])
    plt.show()

#----------------------------------------------------------------------------------------------------

def prediction(test_df, ann_model):
    "This function generates and returns the predictions made by the model."

    test_pred = ann_model.predict(test_df)
    test_pred = np.where(test_pred < 0.5, 0, 1)
    return test_pred

#----------------------------------------------------------------------------------------------------

def print_metric(true_label, pred_label):
  """This function outputs the precision, recall, ROC AUC, and F1 score, representing the model's performance metrics."""

  metrics = pd.DataFrame({"Precision Score":[precision_score(true_label, pred_label)],
                          "Recall Score":[recall_score(true_label, pred_label)],
                          "ROC_AUC Score":[roc_auc_score(true_label, pred_label)],
                          "F1 Score":[f1_score(true_label, pred_label)]})
  print(metrics)

#----------------------------------------------------------------------------------------------------

def save_model(ann_model, path):
    """This function facilitates the preservation of the trained model for future utilization in oil palm image recognition tasks."""

    ann_model.save(path + str(pd.Timestamp.now()) + ".keras")
    print("Model is now saved.")

#----------------------------------------------------------------------------------------------------

def pickle_save_data(data, name_data):
    """This function facilitates the storage of the data into a pickle object.
    - data: a data you want to pickle save.
    - name_data: a string object you want to name the data."""

    return pickle.dump(data, open(name_data+".pickle", "wb"))

#----------------------------------------------------------------------------------------------------

def pickle_load_data(path):
    """This function enables the loading of a pickle object."""

    return pickle.load(open(path, "rb"))

#----------------------------------------------------------------------------------------------------

def load_oilpalm_model(path):
    return load_model(path)