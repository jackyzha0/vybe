# import IPython.display as ipd
# ipd.Audio('../data/Train/2022.wav')

# data, sampling_rate = librosa.load('../data/Train/2022.wav')

# % pylab inline
import os
import pandas as pd
import librosa
import glob 

# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(data, sr=sampling_rate)

# i = random.choice(train.index)

# audio_name = train.ID[i]
# path = os.path.join(data_dir, 'Train', str(audio_name) + '.wav')

# print('Class: ', train.Class[i])
# x, sr = librosa.load('../data/Train/' + str(train.ID[i]) + '.wav')

# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(x, sr=sr)

# train.Class.value_counts()

# _____________________________________________________________________


train=pd.read_csv('C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
train.head()

def parser(row):
  print(row.ID)
  try:
    filename='dataset/Urban/Train/'+str(row.ID)+'.wav'
    x,s=librosa.load(filename,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=x,sr=s,n_mfcc=40).T,axis=0)
  except Exception as e:
    print('Error in ',filename)
    return None,None
  feature=mfccs
  label=row.type
  return [feature,label]

temp=train.apply(parser,axis=1)
temp.columns=['feature','label']


X=np.array(temp.feature.tolist())
Y=np.array(temp.label.tolist())
lb=LabelEncoder()
yy=np_utils.to_categorical(lb.fit_transform(Y))