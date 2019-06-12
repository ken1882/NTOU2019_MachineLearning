import numpy as np 
import pickle

with open('train_div_p0.pkdat', 'rb') as file:
  data = pickle.load(file)

for i in range(10):
  with open('train_div_p{}.pkdat'.format(i+1), 'rb') as file:
    data = np.concatenate((data, pickle.load(file)), axis=0)

print(data.shape)

with open('train.pkdat', 'rb') as file:
  meta = pickle.load(file)
meta['train'] = data
with open('train.pkdat', 'wb') as file:
  pickle.dump(meta, file)