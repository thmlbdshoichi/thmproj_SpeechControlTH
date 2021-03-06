# REAL-TIME KEY-WORD RECOGNITION SPEECH INFERENCE
# ARUNWAT MOONBUNG (SHOICHI)
# IMPORT MODULE ZONE
import argparse
import json
import math
import os
import time
import wave

import alsaaudio
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelBinarizer

import tensorflow as tf
import tensorflow.keras as keras

# ARGPARSER
parser = argparse.ArgumentParser(description='REAL-TIME KEY-WORD RECOGNITION SPEECH INFERENCE (ZCU104 EDITION)')
parser.add_argument('-m','--model', type=str, required=True, help='PATH TO MODEL FILE .h5')
parser.add_argument('-l','--label', type=str, required=True, help='PATH TO LABEL FILE .json')
parser.add_argument('-c','--conf', type=float, required=True, help='SET CONFIDENCE THRESHOLD OF KEYWORD (>0.9)')
parser.add_argument('-dl','--delay', type=float, required=True, help='SET DELAY BETWEEN EACH INFERENCE (sec)')
parser.add_argument('-d','--duration', type=int, required=True, help='SET DURATION OF RECORDING SOUND (1 sec)')
args = parser.parse_args()
# ONE CLASS FOR THE INFERENCE :)

class Keyword_realtime_demo:
    def __init__(self, model_path, text_labels, plot=False, conf=0.7):
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            #self.model.summary()
        else:
            print("!: Unable to Load model, Model directory not found.")
            self.model = None
        if os.path.exists(text_labels):
            with open(text_labels, "r") as f:
                data = json.load(f)
            self.text_labels = [k for k in data.keys()]
        else:
            self.txt_labels = None
        self.plot = plot
        self.conf = conf
    
    def start(self, DURATION):
        # LOAD MFCCs LIST
        # MFCCs = self.record_audio(DURATION=DURATION)
        MFCCs = self.record_audio_alsa(DURATION=DURATION)
        sT = time.time()
        PREDICTED_LABELS = []
        PREDICTED_CONFS = []
        # EXTRACT IT ONE BY ONE
        for idx, MFCC in enumerate(MFCCs):
            # FEED IT TO PREDICT
            #print(f"#: PREDICTED RESULT {idx+1} of {len(MFCCs)}")
            predicted_label, predicted_conf = self.predict(MFCC)
            PREDICTED_LABELS.append(predicted_label)
            PREDICTED_CONFS.append(predicted_conf)
        # MAKE A COMPARE BETWEEN CONF OF EACH LABEL OUTPUT
        FINAL_RESULTS_INDEX = np.argmax(PREDICTED_CONFS)
        FINAL_RESULTS = [PREDICTED_LABELS[FINAL_RESULTS_INDEX], PREDICTED_CONFS[FINAL_RESULTS_INDEX]]
        print(f"R: SUMMARIZE KEYWORDS DETECTED '{FINAL_RESULTS[0]}' with Confidence {FINAL_RESULTS[1]*100:.2f}%\n")
        eT = time.time()
        print(f"INFERENCE USING {eT - sT}")
        return FINAL_RESULTS
    
    def predict(self, MFCCs_INPUT):
        # PREDICT -> OUTPUT PROBABILITY
        predictions = self.model.predict(MFCCs_INPUT)
        predicted_index = np.argmax(predictions)
        predicted_conf = predictions[0][predicted_index]
        predicted_label = self.text_labels[predicted_index]
        predicted_label_fix = self.text_labels[predicted_index]
        if predicted_conf < self.conf:
            predicted_label = "???????????????????????????"
            print(f"!: ERROR TO DETECT KEYWORD | '{predicted_label_fix}' WITH LOW CONFIDENCE {predicted_conf*100:.2f}%")
        else:
            pass
            #print(f"#: KEYWORD DETECTED | '{predicted_label}' WITH CONFIDENCE: {predicted_conf*100:.2f}%")
        return predicted_label, predicted_conf
    
    def preprocess_data_test(self, file_path, DURATION=1, n_mfcc=40, n_fft=4096, hop_length=512, NUM_SAMPLES_TO_CONSIDER=16000):
        # CALCULATION FOR AUDIO FILES
        SAMPLES_PER_TRACK = NUM_SAMPLES_TO_CONSIDER * DURATION
        NUM_SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK/DURATION) # OR REPLACE DURATION WITH NUM_SEGMENTS
        EXPECTED_MFCC = math.ceil(NUM_SAMPLES_PER_SEGMENT / hop_length) # EXPECTED NUMBERS OF MFCCs PER SEGMENT
        # READ AUDIO FILE FROM .WAV FILE TO GET SIGNAL AND SR
        signal, sr = librosa.load(file_path, sr=NUM_SAMPLES_TO_CONSIDER)
        signal = self.pad_audio_sec(signal, DURATION, sr)
        scaler = StandardScaler()
        MFCCs_scaled = []
        for s in range(DURATION): # DURATION = NUM_SEGMENTS
            START_SAMPLE = int(NUM_SAMPLES_PER_SEGMENT * s)
            END_SAMPLE = int(START_SAMPLE + NUM_SAMPLES_PER_SEGMENT)
            MFCC = librosa.feature.mfcc(y=signal[START_SAMPLE:END_SAMPLE],
                                        sr=sr, n_mfcc=n_mfcc,
                                        hop_length=hop_length,n_fft=n_fft)
            MFCC = MFCC.T
            
            if self.plot:
                librosa.display.waveshow(signal[START_SAMPLE:END_SAMPLE], sr=sr)
                #librosa.display.specshow(MFCC, sr=sr, hop_length=hop_length)
                plt.title(f"Audio Signal of {file_path} {s+1} of {DURATION}")
                plt.xlabel("Time (sec)")
                plt.ylabel("Amplitude")
                plt.show()
                
            MFCC_scaled = scaler.fit_transform(MFCC)
            MFCC_scaled = MFCC_scaled.reshape(MFCC_scaled.shape[0], MFCC_scaled.shape[1], 1)
            MFCC_scaled = MFCC_scaled[np.newaxis, ...]
            if len(MFCC) == EXPECTED_MFCC:
                MFCCs_scaled.append(MFCC_scaled)
            else:
                pass
        return MFCCs_scaled
    
    def record_audio_alsa(self, DURATION=1, CHANNELS=1, FORMAT='int16', n_fft=4096, NUM_SAMPLES_TO_CONSIDER=16000, device='default'):
        print(f"####################################################################")
        print(f"#: START RECORDING FOR {DURATION} SEC.. | PLEASE SPEAK KEYWORD NOW!!")
        print(f"####################################################################")
        try:
            with open("cacheSound.raw", "wb") as rf:
                res = []
                recorder = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL,
                channels=CHANNELS, rate=NUM_SAMPLES_TO_CONSIDER, format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=2000, device=device)
                while len(res) < 16000 * DURATION:
                    l, data = recorder.read()
                    a = np.frombuffer(data, dtype='int16')
                    if len(data) != 0:
                        rf.write(a)
                        res.extend(a)
                        time.sleep(.001)
                rf.close()
        except Exception as e:
            print("!: An Error occuring while recording & saving .raw file")
            print(e)
        print(f"#: STOP RECORDING, SAVING CACHE FILE.. SENDING TO AUDIO PRE-PROCESSING")
        try:
            with open("cacheSound.raw", "rb") as inp_f:
                data = inp_f.read()
                with wave.open("cacheSound.wav", "wb") as out_f:
                    out_f.setnchannels(CHANNELS)
                    out_f.setsampwidth(2)
                    out_f.setframerate(NUM_SAMPLES_TO_CONSIDER)
                    out_f.writeframesraw(data)
                    out_f.close()
            inp_f.close()
            return self.preprocess_data_test('cacheSound.wav', DURATION=DURATION)
        except Exception as e:
            print("!: An Error occuring while converting raw file to .wav file")
            print("!: This will occur non keywords detected.")
            return [np.zeros((1, 32, 40, 1))]
    
    def pad_audio(self, signal, NUM_SAMPLES_TO_CONSIDER):
        if len(signal) >= NUM_SAMPLES_TO_CONSIDER:
            return signal[:NUM_SAMPLES_TO_CONSIDER]
        else:
            #return np.pad(signal, pad_width=(0, TOTAL_SAMPLE - len(signal)), mode='constant', constant_values=(0, 0)) # PAD ????????????
            return np.pad(signal, pad_width=(NUM_SAMPLES_TO_CONSIDER - len(signal), 0), mode='constant', constant_values=(0, 0)) # PAD ????????????
    
    def pad_audio_sec(self, signal, DURATION, NUM_SAMPLES_TO_CONSIDER):
        TOTAL_SAMPLE = DURATION*NUM_SAMPLES_TO_CONSIDER
        if len(signal) >= TOTAL_SAMPLE:
            return signal[:TOTAL_SAMPLE]
        else:
            #return np.pad(signal, pad_width=(0, TOTAL_SAMPLE - len(signal)), mode='constant', constant_values=(0, 0)) # PAD ????????????
            return np.pad(signal, pad_width=(TOTAL_SAMPLE - len(signal), 0), mode='constant', constant_values=(0, 0)) # PAD ????????????
        
if __name__ == '__main__':
    print(f"MODEL PATH: {args.model}")
    print(f"MODEL EXISTS: {os.path.exists(args.model)}")
    print(f"LABEL PATH: {args.label}")
    print(f"CONFIDENCE THRESHOLD: {args.conf}")
    print(f"DURATION OF RECORDING: {args.duration} SEC")
    print(f"DELAY: {args.delay} SEC")
    
    KWS_SERVICE = Keyword_realtime_demo(model_path=args.model, text_labels=args.label, conf=args.conf)
    
    try: 
        while True:
            KWS_SERVICE.start(DURATION=args.duration)
            print("")
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("S: ENDING THE REAL-TIME KWS INFERENCE PROCESS")
        print("S: EXIT SYSTEM CODE: USER MANUALLY 'KeyboardInterrupt'")