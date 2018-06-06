from keras.models import load_model
import numpy as np
import keras
import sys



dataset = None

argv = sys.argv

if len(sys.argv) < 4:
    print("Not enough parameters")
    sys.exit(1)
elif len(sys.argv) > 4:
    dataset = argv[4]

model_file, feature_file, label_file = argv[1], argv[2], argv[3]

if dataset:
    f_names = []
    form = 'trn' if dataset == 'train' else 'trl'
    with open('/home/phondanai/src/asv_2017_v2/protocol_V2/ASVspoof2017_V2_{}.{}.txt'.format(dataset, form), 'r') as f:
        for line in f:
            f_names.append(line.split()[0].strip())

# load model from file
model = load_model(model_file)

# load feature from npy file
X = np.load(feature_file)
y = np.load(label_file).ravel()

# expand X dims, get y one-hot
XX = np.expand_dims(X, axis=2)
yy = keras.utils.to_categorical(y - 1, num_classes=2)

score, acc = model.evaluate(XX, yy, batch_size=16)
print('Test score:', score)
print('Test accuracy:', acc)

if dataset:
    answer = {1: 'genuine', 0: 'spoof'}
    preds = model.predict_classes(XX)
    preds_values = model.predict(XX)
    
    for f_name, j ,i in zip(f_names, preds, preds_values):
        print(f_name, answer[j], i[1], i[0], i[1]-i[0])
