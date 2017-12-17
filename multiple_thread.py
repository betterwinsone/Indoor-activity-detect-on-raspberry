# -*- encoding: utf8-*-
#音頻分段使用的library
from pydub import AudioSegment
from pydub.silence import split_on_silence
#mfcc提取使用的library
from speakerfeatures import extract_features
#錄音所使用到的library
from pydub.utils import make_chunks
import wave, pyaudio
from scipy.io.wavfile import read
#多線程library
import threading
#http request library
import requests
import os
#cpikle library
import cPickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 錄音參數設置
CHUNK = 512
FORMAT = pyaudio.paInt16
RATE = 44100
CHANNELS = 1
RECORD_SECONDS = 10
lock = threading.Lock()
speaker_name_len={}#将说话者和说话的时间长度存储成字典{key=name,value=number_of_speaker_times}
activities=[]#活动声音特征
peoplenum=0#室内人数，从server获取
speaker_num=[]#记录每个说话者的时长，len(speaker_num)为说话人数
#开会和报告标记，用于区分活动是开会还是报告
meet=0
report=0
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
    wf.close()
#Segment Function
def filter_segment():
    sound = AudioSegment.from_file("record.wav",format="wav")#讀取音頻文件
    #利用中間沉默的音頻進行分段
    sound = split_on_silence(sound, min_silence_len=700, silence_thresh=-52) #这个函数将每段话一个个分开来，min_silence_len 参数是分割音频时使用的空白声音（每一个音之间会有停顿，用此作为分割的标志）的长度，silence_thresh 是这段空白声音的音量（分贝），分完段之后保存在sound
    new = AudioSegment.empty()
    #将过滤掉静音的音频段合成一段
    for i in sound:
        new =new + i
    new.export('all_row.wav' , format='wav')
    #讀入過濾掉無人說話的聲音的音頻，按照3秒分段
    myaudio = AudioSegment.from_file("all_row.wav" , "wav")
    chunk_length_ms = 3000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    #Export all of the individual chunks as wav files
    chunks_num=len(chunks)
    if chunks is not None:#如果chunks為空，人數為0；
        delete_file()#刪除程序上次分段的chunk，再重新進行分段
        for i, chunk in enumerate(chunks):
            chunk_name = "chunk_wavs/chunk{0}.wav".format(i)
            #print "exporting", chunk_name
            chunk.export(chunk_name, format="wav")
    return chunks_num
#People Count
def count():
    global  speaker_name_len
    global speaker_num
    speaker_name_total=[]
    file_paths=[]
    num=filter_segment()
    if num is 0:
        peoplenum=0
    elif num is 1:
        peoplenum=1
    else:
        for i in range(num-1):#舍弃最后一段
            chunk_name = "chunk_wavs/chunk{0}.wav".format(i)
            file_paths.append(chunk_name)
        modelpath = "speaker_models/" #說話者模型的路徑
        gmm_files = [os.path.join(modelpath,fname) for fname in
                os.listdir(modelpath) if fname.endswith('.pickle')]#讀取訓練好的gmm文件
        models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]#讀取每個gmm文件的內容
        speakers   = [fname.split("/")[-1].split(".pickle")[0] for fname in gmm_files]#讀取說話者標籤
        #人数及说话时长计算
        speaker_num=[]
        for path in file_paths:
            #print file_paths
            path = path.strip()
            sr,audio = read(path)
            vector   = extract_features(audio,sr)
            #计算人数及说话时长
            log_likelihood = np.zeros(len(models))
            for i in range(len(models)):
                gmm    = models[i]         #checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()
            #print speakers,log_likelihood
            winner = np.argmax(log_likelihood)
            print "detected as - ", speakers[winner]
            speaker_name_total.append(speakers[winner])
        #delete repeat speaker
        speaker_name=list(set(speaker_name_total))
        for item in speaker_name:
            speaker_len=speaker_name_total.count(item)*3#每個說話者的長度
            speaker_num.append(speaker_len)
#Activity voice detect function
def activity_detect():
    global activities
    #path to training data
    modelpath = "activity_models/"#活動聲音模型的路徑
    path = "record.wav"#需要分析的音頻
    gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.pickle')]
    #Load the Gaussian gender Models
    models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]
    activity   = [fname.split("/")[-1].split(".pickle")[0] for fname
              in gmm_files]
    # Read the test directory and get the list of test audio files
    path = path.strip()
    sr,audio = read(path)
    vector   = extract_features(audio,sr)#提取mfcc
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    #print speakers,log_likelihood
    winner = np.argmax(log_likelihood)
    print "\tdetected as - ", activity[winner]
    activities=activity[winner]
#delete all chunk*.wav
def delete_file():
    for root, dirs, files in os.walk('chunk_wavs'):
        for name in files:
            if(name.startswith("chunk")):
                os.remove(os.path.join(root, name))
#Detect report
def report_meeting():
    speaker_num.sort()#从小到大排序
    global report,meet
    if len(speaker_num)>=2:
        Max=int(speaker_num[len(speaker_num)-1])#最大的值
        sec_Max=int(speaker_num[len(speaker_num)-2])#第二大的值
    #print Max,sec_Max
        if Max-sec_Max>40:#说话时长最长和说话时长第二长的说话者的时长之差大于40判断为报告，否则是开会状态
            report=1
        else:
            meet=1

#Decide which activities and upload to server
def analysis():
    #室内说话的人数
    global peoplenum,meet,report,speaker_num
    # #从云端获得室内人数
    r = requests.get('http://140.138.152.96:8000/return_peoplenum/')
    return_text=r.text
    peoplenum=str(return_text)
    print "室内人数：",peoplenum
    # #计算说话人数
    count()
    print "说话人数：",len(speaker_num)
    report_meeting()
    #利用各种活动不同的声音特征区分活动
    activity_detect()
    #print "活动：",activities
    #各种活动判断
    if len(speaker_num)==0 and activities=="keyboard" and peoplenum==1:
        print "活动：正在办公"
        #将结果透过http get的方式将最终结果传送到云端
        payload={'people_num':peoplenum,'indoor_statue':"Working"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Upload successfully'
    elif peoplenum>=2 and len(speaker_num)>=1 and report==1:
        print "活动：报告"
        payload={'people_num':peoplenum,'indoor_statue':"Reporting"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Upload successfully'
        report=0
    elif peoplenum>=1 and activities=="clear":
        print "活动:打扫"
        payload={'people_num':peoplenum,'indoor_statue':"Cleaning"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Upload successfully'
    elif peoplenum>=2 and len(speaker_num)>=2 and meet==1:
        print "活动：开会或讨论"
        payload={'people_num':peoplenum,'indoor_statue':"meeting or discussing"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Result upload successfully'
        meet=0
    elif peoplenum>=2 and activities=="music":
        print "活动：聚会"
        payload={'people_num':peoplenum,'indoor_statue':"Have a party"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Result upload successfully'
    elif peoplenum>=2 and activities=="eat":
        print "活动：聚餐"
        payload={'people_num':peoplenum,'indoor_statue':"Have a dinner party"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Result upload successfully'
    elif peoplenum>=1 and activities=="eat":
        print "活动：吃饭中"
        payload={'people_num':peoplenum,'indoor_statue':"Eating"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Result upload successfully'
    elif peoplenum==1 and len(speaker_num)==0 and activities=="environment":
        print "活动：休息"
        payload={'people_num':peoplenum,'indoor_statue':"Resting"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Result upload successfully'
    elif peoplenum==1 and len(speaker_num)==1 and speaker_num[0]>30:
        print "活动：打电话中"
        payload={'people_num':peoplenum,'indoor_statue':"calling"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Result upload successfully'
    else:
        print "活动：其它"
        payload={'people_num':peoplenum,'indoor_statue':"Other activities"}
        re=requests.get('http://140.138.152.96:8000/statues/',params=payload)
        print 'Result upload successfully'
    speaker_num=[]
#thread1
def job1():
    global event
    while True:
        recordWave()
        event.set()

#thread2
def job2():
    global event
    while True:
        event.wait()
        event.clear()
        analysis()

#main function
def main():
    global event
    event=threading.Event()
    t1=threading.Thread(target=job1)
    t2=threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
if __name__ == '__main__':
    main()

