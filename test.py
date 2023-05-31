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
Transformer模型于2017年由Google大脑的一个团队推出，现已逐步取代长短期记忆（LSTM）等RNN模型成为了NLP问题的首选模型。并行化优势允许其在更大的数据集上进行训练。这也促成了BERT、GPT等预训练模型的发展。这些系统使用了维基百科、Common Crawl等大型语料库进行训练，并可以针对特定任务进行微调
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

write_wav("./out/test3.wav", SAMPLE_RATE, piecesCombne)