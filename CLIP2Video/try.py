import pickle 

with open('data/msvd_data/raw-captions.pkl', 'rb') as f:
    a = pickle.load(f)
print(a)
