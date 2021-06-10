from flask import Flask, render_template, request, send_from_directory
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten,Activation
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(112,112,1)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.load_weights('static/eman.h5')
graph = tf.get_default_graph()

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')



@app.route('/home', methods=['POST'])
def home():
    global COUNT
    global graph
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_arr = cv2.resize(img_arr, (112,112))
    img_arr = img_arr / 255.0
    img_arr = np.reshape(img_arr, ( 1,112,112,1))

    #img_arr=np.expand_dims(img_arr, axis=0)


    with graph.as_default():
        prediction = model.predict(img_arr)
    #prediction = model._make_predict_function(img_arr)

    a = round(prediction[0,0], 2)
    b = round(prediction[0,1], 2)
    c = round(prediction[0,2], 2)
    d = round(prediction[0,3], 2)
    e = round(prediction[0,4], 2)
    f = round(prediction[0,5], 2)

    preds = np.array([a,b,c,d,e,f])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)

