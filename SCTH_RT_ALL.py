# REAL-TIME KEY-WORD RECOGNITION SPEECH INFERENCE
# ARUNWAT MOONBUNG (SHOICHI)
# IMPORT MODULE ZONE
import argparse
import json
import math
import os
import pyaudio
import time
import wave

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelBinarizer

import tensorflow as tf
import tensorflow.keras as keras

# ARGPARSER
parser = argparse.ArgumentParser(description='REAL-TIME KEY-WORD RECOGNITION SPEECH INFERENCE')
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
            #self.model.summary() # PRINT MODEL ARCHITECTURE OR NOT?
            time.sleep(3)
        else:
            self.model = None
        if os.path.exists(text_labels):
            with open(text_labels, "r", encoding="utf8") as f:
                data = json.load(f)
            self.text_labels = [k for k in data.keys()]
        else:
            self.txt_labels = None
        self.plot = plot
        self.conf = conf
    
    def start(self, DURATION):
        # LOAD MFCCs LIST
        MFCCs = self.record_audio(DURATION=DURATION)
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
            predicted_label = "ไม่มั่นใจ"
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
        # READ AUDIO FILE FROM PYAUDIO TO GET SIGNAL AND SR
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
                            
            MFCC_scaled = scaler.fit_transform(MFCC)
            MFCC_scaled = MFCC_scaled.reshape(MFCC_scaled.shape[0], MFCC_scaled.shape[1], 1)
            MFCC_scaled = MFCC_scaled[np.newaxis, ...]
            if len(MFCC) == EXPECTED_MFCC:
                MFCCs_scaled.append(MFCC_scaled)
            else:
                pass
        return MFCCs_scaled

    def record_audio(self, DURATION, CHANNELS=1, FORMAT=pyaudio.paInt16, n_fft=4096, NUM_SAMPLES_TO_CONSIDER=16000):
        print(f"####################################################################")
        print(f"#: START RECORDING FOR {DURATION} SEC.. | PLEASE SPEAK KEYWORD NOW!!")
        print(f"####################################################################")
        #time.sleep(0.5) # DELAY IF YOU CAN'T SAY IT IN TIME นะ
        self.pyA = pyaudio.PyAudio()
        self.pyAstream = self.pyA.open(format=FORMAT,channels=CHANNELS,
                                    rate=NUM_SAMPLES_TO_CONSIDER, input=True, output=False,
                                    frames_per_buffer=n_fft)
        frames = [] #int(NUM_SAMPLES_TO_CONSIDER/n_fft * RECORD_SECONDS)
        for i in range(0, int(NUM_SAMPLES_TO_CONSIDER/n_fft * DURATION)): 
            data = self.pyAstream.read(n_fft)
            frames.append(data)
        print("#: STOP RECORDING, SAVING CACHE FILE..")
        #frames_bytes = b''.join(frames)
        #print('Size of Frames_bytes', len(frames_bytes))
        self.pyAstream.stop_stream()
        self.pyAstream.close()
        self.pyA.terminate()
        
        # SAVE RECORDING FILE TO cacheSound.wav TO READ IT
        try:
            wf = wave.open('cacheSound.wav','wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pyA.get_sample_size(FORMAT))
            wf.setframerate(NUM_SAMPLES_TO_CONSIDER)
            wf.writeframes(b''.join(frames))
            wf.close()
            return self.preprocess_data_test('cacheSound.wav', DURATION=DURATION)
        except:
            print("!: An Error occuring while saving .wav file")
            print("!: This will occur non keywords detected.")
            return [np.zeros((1, 32, 40, 1))]
        
    def pad_audio(self, signal, NUM_SAMPLES_TO_CONSIDER):
        if len(signal) >= NUM_SAMPLES_TO_CONSIDER:
            return signal[:NUM_SAMPLES_TO_CONSIDER]
        else:
            #return np.pad(signal, pad_width=(0, TOTAL_SAMPLE - len(signal)), mode='constant', constant_values=(0, 0)) # PAD หลัง
            return np.pad(signal, pad_width=(NUM_SAMPLES_TO_CONSIDER - len(signal), 0), mode='constant', constant_values=(0, 0)) # PAD หน้า
    
    def pad_audio_sec(self, signal, DURATION, NUM_SAMPLES_TO_CONSIDER):
        TOTAL_SAMPLE = DURATION*NUM_SAMPLES_TO_CONSIDER
        if len(signal) >= TOTAL_SAMPLE:
            return signal[:TOTAL_SAMPLE]
        else:
            #return np.pad(signal, pad_width=(0, TOTAL_SAMPLE - len(signal)), mode='constant', constant_values=(0, 0)) # PAD หลัง
            return np.pad(signal, pad_width=(TOTAL_SAMPLE - len(signal), 0), mode='constant', constant_values=(0, 0)) # PAD หน้า

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
