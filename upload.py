#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'pkl'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    import pprint, pickle
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.metrics import Accuracy
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join("uploads", filename))
            
            pkl_file = open('uploads\data.pkl', 'rb')
            X_test_norm = pickle.load(pkl_file)
            pkl_file.close()

            y_test_cat = np.load('y_test_cat.npy')

            myModel = load_model('pneu')
            predictions2 = myModel.predict(X_test_norm)

            m = Accuracy()
            m.update_state(y_test_cat, np.round(predictions2))
            print('The Accuracy is', m.result().numpy())
            ans = m.result().numpy()
            return f'The Prediction Accuracy is: {ans} and the Predictions are {predictions2}' #redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

