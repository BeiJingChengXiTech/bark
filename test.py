import os
from scipy.io.wavfile import write as write_wav

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE


preload_models()

script = """
Hey, have you heard about this new text-to-audio model called "Bark"? 
Apparently, it's the most realistic and natural-sounding text-to-audio model 
out there right now. People are saying it sounds just like a real person speaking. 
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)

SPEAKER = "v2/en_speaker_6"
speaker_lookup = ["v2/en_speaker_9","v2/en_speaker_2"]

silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
piecesCombne = np.zeros(int(0 * SAMPLE_RATE))
num = 0
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=speaker_lookup[num%2])
    num += 1
    piecesCombne = np.append(piecesCombne, audio_array)
    # write pieces to wav file
    # format string "./out/test"+num+".wav"

write_wav("./out/test"+str(num)+".wav", SAMPLE_RATE, piecesCombne)