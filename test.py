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
Transformer模型（直译为“变换器”）是一种采用自注意力机制的深度学习模型，这一机制可以按输入数据各部分重要性的不同而分配不同的权重。该模型主要用于自然语言处理（NLP）与电脑视觉（CV）领域。
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)

SPEAKER = "v2/zh_speaker_9"

silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
piecesCombne = np.zeros(int(0 * SAMPLE_RATE))
num = 0
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    num += 1
    piecesCombne = np.append(piecesCombne, audio_array)
    # write pieces to wav file
    # format string "./out/test"+num+".wav"

write_wav("./out/test"+str(num)+".wav", SAMPLE_RATE, piecesCombne)