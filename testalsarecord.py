import wave
import alsaaudio
import time
import numpy as np
from playsound import playsound

def record_audio_alsa(DURATION=1, CHANNELS=1, FORMAT='int16', n_fft=4096, NUM_SAMPLES_TO_CONSIDER=16000, device='default'):
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
        print(f"#: STOP RECORDING, SAVING AUDIO .RAW FILE..")
        print(f"#: CONVERTING AUDIO RAW FILE -> .WAV FILE..")
        
        try:
            with open("cacheSound.raw", "rb") as inp_f:
                data = inp_f.read()
                with wave.open("cacheSound.wav", "wb") as out_f:
                    out_f.setnchannels(CHANNELS)
                    out_f.setsampwidth(2)
                    out_f.setframerate(NUM_SAMPLES_TO_CONSIDER)
                    out_f.writeframesraw(data)
                    out_f.close()
                    print(f"#: WAV FILES HAS BEEN CONVERTED, SENDING TO AUDIO PREPROCESSING.")
                    print(f"####################################################################")
            inp_f.close()
            return "FINISH RECORDING"
            #return self.preprocess_data_test('cacheSound.wav', DURATION=DURATION)
        except Exception as e:
            print("!: An Error occuring while converting raw file to .wav file")
            print("!: This will occur non keywords detected.")
            return "ERROR RECORDING"
        
if __name__ == '__main__':
    record_audio_alsa(DURATION=5)
    print("NOW PLAYING: cacheSound.wav")
    playsound('cacheSound.wav')