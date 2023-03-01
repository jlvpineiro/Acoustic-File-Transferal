import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,fftfreq,ifft
import binascii
from scipy.io import wavfile
from bisect import insort
import scipy.signal as sig
import random
import numpy.random as rnd
import time
import pyaudio
from cryptography.fernet import Fernet

dur = 0.1
syncDur = 1.0
Fs = 44100
fsync = 3000
compression = 1
if compression < 5:
    fbase = 200
else:
    fbase = 200
msbit = 4



# import pyaudio
# import sounddevice as sd


# import alsaaudio as audio
# import time
# import audioop

# #Input & Output Settings
# periodsize = 1024
# audioformat = audio.PCM_FORMAT_S16_LE
# channels = 1
# framerate=Fs

# #Input Device
# inp = audio.PCM(audio.PCM_CAPTURE,audio.PCM_NONBLOCK)
# inp.setchannels(channels)
# inp.setrate(framerate)
# inp.setformat(audioformat)
# inp.setperiodsize(periodsize)

# #Output Device
# out = audio.PCM(audio.PCM_PLAYBACK)
# out.setchannels(channels)
# out.setrate(framerate)
# out.setformat(audioformat)
# out.setperiodsize(periodsize)

def fileToHex(filename):
    hexrep = ''
    with open(filename, 'rb') as f:
        content = f.read()
        hexrep += str(binascii.hexlify(content))
    return hexrep[2:-1]

def hexToBin(hexstr,pad='0',rjust='>',kind='b'):
    hexnum = int(hexstr,16)
    size = 4*len(hexstr)
    return f'{hexnum:{pad}{rjust}{size}{kind}}'

def hexToAudio(hexstr,key='',dur=dur,compression=compression,fbase=fbase,msbit=msbit,Fs=Fs,doSync=True):
    hexlen = len(hexstr)
    keylen = len(key)
    if hexlen > int('F'*msbit,16):
        return "file is too big"
    t = np.linspace(0,dur,int(dur*Fs),endpoint=False)
    silence = np.zeros(int(Fs/4))
    hexHexLen = "{0:0{1}x}".format(hexlen,msbit)
    hexKeyLen = "{0:0{1}x}".format(keylen,msbit)
    hexstr = hexHexLen + hexKeyLen + hexstr + key
    hexLenSig = encodeLength(hexHexLen,msbit,t)
    keyLenSig = encodeLength(hexKeyLen,msbit,t)
    audio = np.concatenate((hexLenSig,keyLenSig))
    start = 2*msbit

    for i in range(start,len(hexstr),compression):
        tone = np.zeros(len(t))
        ind=0
        while ind < compression and i + ind < len(hexstr):
            f = fbase + 50 * int(hexstr[i+ind],16)
            f += 1000*(ind+1)
            tone += 30*np.cos(2*np.pi*f*t)
            ind+=1
        audio = np.concatenate((audio,tone,silence))
    if doSync:
        audio = addSyncBit(audio)
    return audio

def encodeLength(n,msbit,t,fbase=fbase):
    lenTone = np.zeros(0)
    silence = np.zeros(int(Fs/4))
    for i in range(0,msbit,compression):
        ind = 0
        tone = np.zeros(len(t))
        while ind < compression and i + ind < msbit:
            f = fbase + 50 * int(n[i+ind],16)
            f += 1000*(ind+1)
            print(n[i+ind],f)
            tone += 30*np.cos(2*np.pi*f*t)
            ind+=1
        lenTone = np.concatenate((lenTone,tone,silence))
    return lenTone

def fileToAudio(filename,doEncrypt=False,dur=dur,compression=compression,fbase=fbase,msbit=msbit,Fs=Fs):
    if doEncrypt:
        key = Fernet.generate_key()
        with open(filename+'.key','wb') as f:
            f.write(key)
            f.close()
        with open(filename,'rb') as f:
            fdata = f.read()
        fernet = Fernet(key)
        encrypted = fernet.encrypt(fdata)
        with open(filename+'.encrypted','wb') as f:
            f.write(encrypted)
        inFile = filename+'.encrypted'
        keystr = fileToHex(filename+'.key')
    else:
        inFile = filename
        keystr = ''
    hexstr = fileToHex(inFile)
    signal = hexToAudio(hexstr,key=keystr,dur=dur,compression=compression,fbase=fbase,msbit=msbit,Fs=Fs)
    return signal

def binToAudio(binstr,dur=0.5,fplay=440,Fs=Fs):
    t = np.linspace(0,dur,int(dur*Fs),endpoint=False)
    audio = []
    for b in binstr:
        audio.extend(int(b) * np.cos(2*np.pi*fplay*t))
    return audio

def writeFileToWav(filename,outputFile='output.wav',dur=dur,compression=compression,fbase=fbase,msbit=msbit,Fs=Fs):
    signal = fileToAudio(filename,dur,compression,fbase,msbit,Fs)
    wavfile.write(outputFile,Fs,signal)
    return

def writeSignalToWav(signal,outputFile='output.wav',Fs=Fs):
    wavfile.write(outputFile,Fs,signal)
    return

def readWav(filename):
    signal = wavfile.read(filename)[1]
    return signal

def writeSignalToFile(signal,outputFile='decoded_out',dur=dur,compression=compression,fbase=fbase,msbit=msbit,Fs=Fs):
    decoded = decode(signal,dur,compression,fbase,msbit,Fs)
    encrypted = ',' in decoded
    if encrypted:
        seperator = decoded.index(',')
        hexstr = decoded[:seperator]
        keystr = decoded[seperator+1:]
    else:
        hexstr = decoded

    with open(outputFile,'wb') as fout:
        if encrypted:
            with open(outputFile+'.encrypted','wb') as temp:
                temp.write(binascii.unhexlify(hexstr))
            with open(outputFile+'.encrypted','rb') as temp:
                data = temp.read()
            with open(outputFile+'.key','wb') as keyf:
                key = binascii.unhexlify(keystr)
                keyf.write(key)
            fernet = Fernet(key)
            outdata = binascii.hexlify(fernet.decrypt(data))
        else:
            outdata = hexstr
        fout.write(binascii.unhexlify(outdata))
    return

# def butter_bandstop_filter(signal,f0,Fs=Fs,Q=30):
    # nyq = 0.5 * Fs
    # w0 = f0 / nyq

    # b,a = sig.iirnotch(w0,Q,Fs)
    # y = sig.lfilter(b,a,signal)
    # return y


def decode(signal,dur=dur,compression=compression,fbase=fbase,msbit=msbit,Fs=Fs):
    hexstr = ''
    start = syncRT(signal)
    inc = int(Fs/4) + int(dur*Fs)
    signal = signal[start-inc//2:]
    for i in range(2):
        count = msbit
        nFreqs = []
        while count > 0:
            segment = signal[:inc]
            if count >= compression:
                newfreqs = extractFreqs(segment,compression,graph=False)
                for f in newfreqs:
                    nFreqs.append(f)
                count -= compression
            else:
                newfreqs = extractFreqs(segment,count,graph=False)
                for f in newfreqs:
                    nFreqs.append(f)
                count = 0
            if count > 0:
                signal = signal[inc:]
        if i == 0:
            hexLen = int(freqsToHex(nFreqs),16)
            signal = signal[inc:]
        else:
            keyLen = int(freqsToHex(nFreqs),16)
    print("hexLen = " + str(hexLen) + ", keyLen = " + str(keyLen))
    hexStart = inc+inc//2
    for i in range(2):
        if i == 0:
            n = hexLen
        else:
            if keyLen == 0:
                continue
            else:
                n = keyLen
        for j in range(hexStart,len(signal),inc):
            segment = signal[j-inc//2:j+inc//2]
            hexStart += inc
            if (n >= compression):
                segfreqs = extractFreqs(segment,compression,graph=False)
                n -= compression
                # segfreqs = []
                # sigft = fft(segment,n=fs)
                # sigft = abs(sigft[:len(sigft)//2])
                # for k in range(compression):
                    # # sigft = fft(segment,n=fs)
                    # # # sigft = sigft[range(int(fs/2))]
                    # # # sigftmag = abs(sigft)
                    # # freqs = fftfreq(len(sigft))
                    # freqidx = int(np.argmax(abs(sigft))*(fs/(sigft.size*2)))
                    # # maxfreq = freqs[freqidx]
                    # # freqhz = abs(maxfreq*fs)
                    # # print(freqidx)
                    # sigft[freqidx] = 0
                    # # maxfreq = np.argmax(sigftmag)*(fs/(sigft.size*2))
                    # insort(segfreqs,int(freqidx))
                    # # segment = butter_bandstop_filter(segment,maxfreq)
            else:
                segfreqs = extractFreqs(segment,n)
                n = 0
            if segfreqs:
                hexstr += freqsToHex(segfreqs)
            if n <= 0:
                break
        if i == 0:
            hexstr += ','
    return hexstr

def extractFreqs(signal,numFreqs,doRounding=True,ftol=15,Fs=Fs,syncing=False,decoding=True,graph=False):
    # print("numFreqs = " + str(numFreqs))
    if numFreqs > 0:
        segFreqs = []
        sigft = fft(signal,n=Fs)
        sigft = abs(sigft[:len(sigft)//2])
        if decoding:
            sigft[:fbase] = 0
        for j in range(numFreqs):
            # sigft = fft(segment,n=Fs)
            # # sigft = sigft[range(int(Fs/2))]
            # # sigftmag = abs(sigft)
            # freqs = fftfreq(len(sigft))
            if syncing:
                sigft[:500] = 0
            if graph:
                plt.plot(sigft)
                plt.show()
            freqidx = int(np.argmax(abs(sigft))*(Fs/(sigft.size*2)))
            if doRounding:
                freqidx = roundFreq(freqidx)
            sigft[freqidx-ftol:freqidx+ftol] = 0
            if decoding:
                if len(freqsToHex([freqidx])) < 2:
                    # print(freqsToHex([freqidx]))
                    asciivalue = ord(freqsToHex([freqidx]))
                    if (asciivalue >= 48 and asciivalue <= 57) or (asciivalue >= 97 and asciivalue <= 102):
                        insort(segFreqs,int(freqidx))
                    else:
                        j-=1
                else:
                    j-=1
            else:
                # print(freqidx)
                insort(segFreqs,int(freqidx))
            # maxfreq = freqs[freqidx]
            # freqHz = abs(maxfreq*Fs)
            # print(freqidx)
            # maxfreq = np.argmax(sigftmag)*(Fs/(sigft.size*2))
            # segment = butter_bandstop_filter(segment,maxfreq)

        return segFreqs

def freqsToHex(freqs,fbase=fbase):
    hexstr = ''
    for f in freqs:
        # print(f)
        f = f % 1000
        f = int((f-fbase)/50)
        hexchar = hex(f)[2:]
        hexstr+=hexchar
    return hexstr


def roundFreq(f):
    fthous = int(f/1000)
    f = f%1000
    f = round((f-fbase)/50)
    f = fbase + 50 * f
    return fthous*1000 + f

def syncRT2(signal,fsync=fsync,Fs=Fs):
    syncing = False
    synced = False
    start = 0
    end = start+1
    while not syncing or not synced:
        end+=1
        if syncing:
            start+=1
        ft = fft(signal[start:end])
        ft = ft[:int(len(ft)/2)]
        freqidx = int(np.argmax(abs(ft))*(Fs/(ft.size*2)))
        # print(freqidx)
        synced = syncing and (fsync != freqidx)
        if not synced:
            syncing = (fsync == freqidx)  
    return end - 133

def syncRT(signal,dur=syncDur,fsync=fsync,ftol=15,chunkSize=1028,syncTimes=1,compression=compression,Fs=Fs):
    start = 0
    # syncVol = compression*3
    # avg = np.average(abs(signal))
    phase = 0
    while phase < 7:
        # print(phase)
        chunkFreq = extractFreqs(signal[start:start+chunkSize],1,syncing=True,decoding=False,doRounding=False,graph=False)[0]
        # print(chunkFreq)
        if chunkFreq > fsync - ftol and chunkFreq < fsync + ftol: 
            if phase == 0:
                phase += 1
        elif phase >= 1 and (chunkFreq < fsync - ftol or chunkFreq > fsync + ftol):
            phase += 1
        start += chunkSize
    start -= (phase-2) * chunkSize
    start += int(dur*Fs)

    # while signal[start] >= syncVol - avg:
        # start+=1
        # if start >= len(signal)-1:
            # return 0
    # print(start+1)
    print(start)
    return start

# def syncRT(signal,dur=dur,fsync=fsync,syncTimes=1,compression=compression,Fs=Fs):
    # start = 0
    # syncVol = compression*3
    # avg = np.average(abs(signal))
    # while signal[start] < syncVol - avg:
        # start+=1
        # if start >= len(signal)-1:
            # return 0
    # start += int(dur*Fs)*syncTimes
    # # while signal[start] >= syncVol - avg:
        # # start+=1
        # # if start >= len(signal)-1:
            # # return 0
    # print(start+1)
    # return start+1

def addSyncBit(signal,dur=syncDur,syncTimes=1,fsync=fsync,compression=compression,Fs=Fs):
    syncVol = 30
    t = np.linspace(0,dur,int(dur*Fs))
    syncTone = syncVol*np.cos(2*np.pi*fsync*t)
    syncSig = np.zeros(0)
    silence = np.zeros(len(t))
    for i in range(syncTimes):
        syncSig = np.concatenate((syncSig,syncTone))
    signal  = np.concatenate((syncSig,silence,signal))
    signal  = np.concatenate((signal,silence,syncSig))
    return signal

# def testSync2():
    # signal = fileToAudio('test.txt')
    # t = np.linspace(dur,Fs*2)
    # syncsig = np.cos(np.pi*2*t*12000)
    # signal = np.concatenate((syncsig,signal))
    # a = syncRT2(signal)
    # return a

# def testSync(testfile,sigma=1.0):
    # signal = fileToAudio(testfile)
    # signal = addSyncBit(signal)
    # silence = np.zeros(random.randint(0,10000))
    # signalSil  = np.concatenate((silence,signal))
    # count = 0
    # noise = sigma*rnd.randn(len(signalSil))
    # noisySignal = signalSil + noise
    # hexstr = fileToHex(testfile)
    
    # while decode(noisySignal) == hexstr and count < 100:
        # print(count)
        # print(sigma)
        # count+=1
        # sigma+=0.1
        # silence = np.zeros(random.randint(0,10000))
        # signalSil  = np.concatenate((silence,signal))
        # noise = sigma*rnd.randn(len(signalSil))
        # noisySignal = signalSil + noise
    # return count

def record(Fs=Fs,fsync=fsync,FORMAT=pyaudio.paInt16,CHANNELS=1,CHUNK=1024,minCount=10):
    """ Gets average audio intensity of your mic sound. You can use it to get
        average intensities while you're talking and/or silent. The average
        is the avg of the 20% largest intensities recorded.
    """

    print("Getting intensity values from mic.")
    p = pyaudio.PyAudio()
    theaudio = np.zeros(0)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=Fs,
                    input=True,
                    frames_per_buffer=CHUNK)

    # values = [math.sqrt(abs(audioop.avg(stream.read(CHUNK), 4))) 
              # for x in range(num_samples)] 
    ftol = 15
    flag = False
    count = 0
    phase = 0
    while phase < 4:
        data = stream.read(CHUNK)
        numpydata = np.frombuffer(data,dtype=np.int16)
        ft = np.fft.fft(numpydata,n=Fs)
        ft = abs(ft[:len(ft)//2])
        ft[:500] = 0
        freqidx = int(np.argmax(abs(ft))*(Fs/(ft.size*2)))

        print(freqidx)
        if freqidx > fsync - ftol and freqidx < fsync + ftol:
            if not flag:
                flag = True
                phase += 1
                print("phase = " + str(phase))
            if flag:
                count+=1
        elif flag and (freqidx < fsync - ftol or freqidx > fsync + ftol):
            flag = False
            if count >= minCount:
                phase += 1
                print("phase = " + str(phase))
            else:
                phase -= 1
                print("phase = " + str(phase))
                if phase < 1:
                    theaudio = np.zeros(0)
            count = 0

        if phase > 0:
            theaudio = np.concatenate((theaudio,numpydata))


        # theaudio = np.append(theaudio,np.fromstring(stream.read(CHUNK),dtype=np.float64))
    # values = sorted(values, reverse=True)
    # r = sum(values[:int(num_samples * 0.2)]) / int(num_samples * 0.2)
    print(" Finished ")
    # print(" Average audio intensity is ", r)
    stream.close()
    p.terminate()
    # audioft = fft(theaudio)
    # audioftfilt = audioft * (abs(audioft) > 500)
    # theaudio = ifft(audioftfilt)
    return theaudio

# def testMic():
    # mic_name = 'default'
    # sample_rate = Fs
    # chunk_size=2048
    # r = sr.Recognizer()
    # mic_list = sr.Microphone.list_microphone_names()

    # #the following loop aims to set the device ID of the mic that
    # #we specifically want to use to avoid ambiguity.
    # for i, microphone_name in enumerate(mic_list):
        # if microphone_name == mic_name:
            # device_id = i

    # with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                        # chunk_size = chunk_size) as source: 
        # r.adjust_for_ambient_noise(source)
        # print("Say something")
        # audio = r.listen(source)
        # print(audio)


# def callback(in_data, frame_count, time_info, flag):
    # # using Numpy to convert to array for processing
    # audio_data = np.fromstring(in_data, dtype=np.float32)
    # total_audio = np.concatenate((total_audio,audio_data))
    # print(max(audio_data))
    # return in_data, pyaudio.paContinue

# def testMic(CHANNELS=1,CHUNK=1024,RATE=Fs):
    # p = pyaudio.PyAudio()
    # total_audio = np.zeros(0)



    # stream = p.open(format=pyaudio.paFloat32,
                    # channels=CHANNELS,
                    # rate=RATE,
                    # # output=True,
                    # input=True,
                    # frames_per_buffer=CHUNK,
                    # stream_callback=callback)

    # stream.start_stream()

    # while stream.is_active():
        # time.sleep(20)
        # stream.stop_stream()
        # print("Stream is stopped")

    # stream.close()

    # p.terminate()
    # return writeSignalToWav(total_audio,outputFile='micTest.wav')

# def testMic(Fs=Fs,outputFile='recording.wav'):
    # seconds = 3  # Duration of recording

    # myrecording = sd.rec(int(seconds * Fs), samplerate=Fs, channels=2)
    # # for i in range(100):
        # # print(np.amax(myrecording))
    # sd.wait()  # Wait until recording is finished
    # wavfile.write(outputFile, Fs, myrecording)  # Save as WAV file 

testsig = fileToAudio('test.txt')
