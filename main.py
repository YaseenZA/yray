from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def index():
    print('loading packages')
    import pprint, pickle
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.metrics import Accuracy
    
    pkl_file = open('data.pkl', 'rb')
    X_test_norm = pickle.load(pkl_file)
    pkl_file.close()

    y_test_cat = np.load('y_test_cat.npy')

    myModel = load_model('pneu')
    predictions2 = myModel.predict(X_test_norm)

    m = Accuracy()
    m.update_state(y_test_cat, np.round(predictions2))
    print('The Accuracy is', m.result().numpy())
    ans = m.update_state(y_test_cat, np.round(predictions2))
    return str(ans)


@app.route('/index', methods = ["GET", "POST"])
def form():
    print('these are the file in the directory!!')
    import os
    arr = os.listdir('.')
    print(arr)
    return render_template('index.html')

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True, port=5000)

    
 