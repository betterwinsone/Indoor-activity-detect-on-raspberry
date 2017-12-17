# -*- encoding: utf8-*-
from pydub import AudioSegment
from pydub.silence import split_on_silence

from scipy import spatial

from pydub.utils import make_chunks
import wave, pyaudio
import threading
import time
import numpy as np


#test_gender.py
import os
import cPickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import scipy.io.wavfile as wav
# Settings
CHUNK = 512
FORMAT = pyaudio.paInt16
RATE = 44100
CHANNELS = 1
RECORD_SECONDS = 10
flag=0
lock = threading.Lock()
# Record Function
def recordWave():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)
    print 'Recording...'
    buffer = []
    for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
        audio_data = stream.read(CHUNK)
        buffer.append(audio_data)
    print 'Record Done'
    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf = wave.open('record.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(buffer))
#    print "================= %d " % len(buffer)
    wf.close()
def filter_segment(dd):
    file_chunk_name="record_wavs/record{0}.wav".format(dd)
    sound = AudioSegment.from_file(file_chunk_name,format="wav")#读取音频文件
    #利用中间的沉默声音进行音频分段
    sound = split_on_silence(sound, min_silence_len=700, silence_thresh=-50) #这个函数将每段话一个个分开来，min_silence_len 参数是分割音频时使用的空白声音（每一个音之间会有停顿，用此作为分割的标志）的长度，silence_thresh 是这段空白声音的音量（分贝），分完段之后保存在sound
    new = AudioSegment.empty()
    #将过滤掉静音的音频段合成一段
    for i in sound:
        new =new + i
    new.export('all_row.wav' , format='wav')
    #读入过滤静音后的音频，然后按照1秒分段
    myaudio = AudioSegment.from_file("all_row.wav" , "wav")
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    #Export all of the individual chunks as wav files
    chunks_num=len(chunks)
    if chunks is not None:#如果chunks为空，人数为0；
        delete_file()
        for i, chunk in enumerate(chunks):
            chunk_name = "chunk_wavs/chunk{0}.wav".format(i)
            #print "exporting", chunk_name
            chunk.export(chunk_name, format="wav")
    return chunks_num
def computing_mfcc(chunk_name):
    dataset=[]
    mfcc_list=[]
    (rate ,sig ) = wav .read (chunk_name)  #提取音频
    mfcc_feat =  mfcc.mfcc(sig,samplerate=rate,winlen=0.025,winstep=0.01,numcep=13,nfft=1103,appendEnergy=True)#输出的值是一二维数组，行代表的是一段音频分成多少个时间段，列代表的是同一个时间的倒频系数的个数，参数请看https://github.com/jameslyons/python_speech_features
    #mfcc_feat = preprocessing.scale(mfcc_feat)
    for i in range(len(mfcc_feat)):
        for i1 in mfcc_feat[i][1:20]:#取出mfcc_feat1 index 1-19的值，去除了index1的值
            dataset.append(i1)
        mfcc_list.append(dataset)
        dataset=[]
    return mfcc_list

def compare_mfcc(mfcc1,mfcc2):
    sum=0.0
    c=0
    for j in range(len(mfcc1)):
        cosine_distance = spatial.distance.cosine(mfcc1[j], mfcc2[j])#distance"1-cos"越小越相似
        sum=sum+cosine_distance
        c=c+1
        ave_cs=sum/c
    return ave_cs

def count(ii):
    all_mfcc=[]
    new_speaker=[]
    num=filter_segment(ii)
    if num is 0:
        print "人数为0"
    elif num is 1:
        print "人数为1"
    else:
        for i in range(num-1):#舍弃最后一段
            chunk_name = "chunk_wavs\\chunk{0}.wav".format(i)
            mfcc=computing_mfcc(chunk_name)
            all_mfcc.append(mfcc)
        k=1
        c=0
        sum=0
        flag=0
        new_speaker.append(all_mfcc[0])
        for j in range(1,len(all_mfcc)):#6段音频
            ave=[]
            for j1 in range(k):#新增的音频
                dist_avg=compare_mfcc(new_speaker[j1], all_mfcc[j])
                print j,j1,dist_avg
                ave.append(dist_avg)
            #print ave
            k1=0
            for i in range(k):
                if ave[i] > 0.69:
                    k1=k1+1
                else:
                    break;
            if k==k1:
                new_speaker.append(all_mfcc[j])
                k=k+1
        print k

def delete_file():
    for root, dirs, files in os.walk('chunk_wavs'):
        for name in files:
            if(name.startswith("chunk")):
                os.remove(os.path.join(root, name))

def compare_all_dist():
    all_mfcc=[]
    new_speaker=[]
    num=filter_segment()
    if num is 0:
        print "人数为0"
    elif num is 1:
        print "人数为1"
    else:
        for i in range(num-1):#舍弃最后一段
            chunk_name = "chunk_wavs\\chunk{0}.wav".format(i)
            mfcc=computing_mfcc(chunk_name)
            all_mfcc.append(mfcc)

    print "所有mfcc的长度：",len(all_mfcc)
    print "不相同的音频段及余弦距离："
    for i in range(4):
        for j in range(5,len(all_mfcc)):
            if i is j:
                continue
            dist_avg=compare_mfcc(all_mfcc[i],all_mfcc[j])
            print "音频段：",i,"音频段：",j,"余弦距离：",dist_avg

    print "相同的音频段及余弦距离："
    # for i in range(3):
    #     for j in range(i,3):
    #         if i is j:
    #             continue
    #         dist_avg=compare_mfcc(all_mfcc[i],all_mfcc[j])
    #         print "音频段：",i,"音频段：",j,"余弦距离：",dist_avg
    for i in range(5,9):
        for j in range(i,9):
            if i is j:
                continue
            dist_avg=compare_mfcc(all_mfcc[i],all_mfcc[j])
            print "音频段：",i,"音频段：",j,"余弦距离：",dist_avg

def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,samplerate=sr,winlen=0.025,winstep=0.01,numcep=13,nfft=1103,appendEnergy=True)
    feat     = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i,:]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat;
    features = preprocessing.scale(features)
    return features

def Indect_Act(ii):
    #path to saved models
    modelpath  = "Activity/gmm_file"
    gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.pickle')]
    models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]#load gmm file
    activity_name   = [fname.split("\\")[-1].split(".pickle")[0] for fname in gmm_files]#save gmm filename as label
    #file to testing data
    chunk_name="record_wavs/record{0}.wav".format(ii)
    #(rate ,sig ) = wav .read (chunk_name)
    #print f.split("\\")[-1]
    #加锁
    sr, audio  = wav .read (chunk_name)
    features   = get_MFCC(sr,audio)
    scores     = None
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(features))
        log_likelihood[i] = scores.sum()
    winner = np.argmax(log_likelihood)
    print "detected as - ", activity_name [winner]
    print "dsadas"
    # print "\n\tscores:"
    # for i in range(len(activity_name)):
    #     print "\t",activity_name[i] ,log_likelihood[i]
#线程
def record_loop():
    while True:
        eve.clear()
        recordWave()
	#time.sleep(3)
        eve.set()


def count_loop():
    while True:
        #eve.wait()
        print "start count:"
        count()
        time.sleep(3)

def Activity_loop():
    while True:
        #eve.wait()
        print "start act:"
        Indect_Act()
        time.sleep(3)

def main():
    time.sleep(60)
    global eve
    eve = threading.Event()
    lock = threading.Lock()
    #record_thread=threading.Thread(target=record_loop)#add record thread
    count_thread=threading.Thread(target=count_loop)#add count thread
    Activity_thread=threading.Thread(target=Activity_loop)#add Activity thread
    #record_thread.start()
    #count_thread.start()
    #Activity_thread.start()
    ii=0
    while True:
        count(ii)
	Indect_Act(ii)
        ii=ii+1
if __name__ == '__main__':
    main()
