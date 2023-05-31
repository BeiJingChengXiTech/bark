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
与循环神经网络（RNN）一样，Transformer模型旨在处理自然语言等顺序输入数据，可应用于翻译、文本摘要等任务。而与RNN不同的是，Transformer模型能够一次性处理所有输入数据。注意力机制可以为输入序列中的任意位置提供上下文。如果输入数据是自然语言，则Transformer不必像RNN一样一次只处理一个单词，这种架构允许更多的并行计算，并以此减少训练时间。
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

write_wav("./out/test2.wav", SAMPLE_RATE, piecesCombne)