from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('eman2.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')

classes = {1:'buildings',
           2:'forest',
           3:'glacier',
           4: 'mountain',
           5: 'sea',
           6: 'street'}



@app.route('/home', methods=['POST'])
def home():
    global COUNT

    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    img_arr = Image.open('static/{}.jpg'.format(COUNT))

    img = np.resize(img_arr, (112, 112))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    normalized = img
    reshaped = np.reshape(normalized, (1, 112, 112, 1))
    mg_arr = np.array(reshaped)
    result = model.predict_classes(mg_arr)[0]

    sign = classes[result +1]
    COUNT += 1

    return render_template('index.html', prediction=sign)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)

