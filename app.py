#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
from flask import Flask, flash, request, redirect, url_for, render_template
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
    from PIL import Image
    
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
            
            filedata = np.array(Image.open(os.path.join("uploads", filename)).convert('L').resize((132, 97)))
            
            myModel = load_model('pneu')
            X_test_norm = np.round((filedata/255), 3).copy()
            print('before reshaping', X_test_norm.shape)
            X_test_norm = X_test_norm.reshape(-1, 97, 132, 1)
            print('after reshaping', X_test_norm.shape)
            
            predictions2 = myModel.predict(X_test_norm)

            return render_template('submitted.html', predictions2=predictions2, filename=filename)#f'The predicted likelihood is: {predictions2}' #redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


# In[ ]:




