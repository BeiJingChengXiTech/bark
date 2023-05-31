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

与循环神经网络（RNN）一样，Transformer模型旨在处理自然语言等顺序输入数据，可应用于翻译、文本摘要等任务。而与RNN不同的是，Transformer模型能够一次性处理所有输入数据。注意力机制可以为输入序列中的任意位置提供上下文。如果输入数据是自然语言，则Transformer不必像RNN一样一次只处理一个单词，这种架构允许更多的并行计算，并以此减少训练时间。

Transformer模型于2017年由Google大脑的一个团队推出，现已逐步取代长短期记忆（LSTM）等RNN模型成为了NLP问题的首选模型。并行化优势允许其在更大的数据集上进行训练。这也促成了BERT、GPT等预训练模型的发展。这些系统使用了维基百科、Common Crawl等大型语料库进行训练，并可以针对特定任务进行微调

在Transformer模型之前，大多数最先进的NLP系统都依赖于诸如LSTM、门控循环单元（GRU）等门控RNN模型，并在此基础上增加了注意力机制。Transformer正是在注意力机制的基础上构建的，但其没有使用RNN结构，这表明仅依靠注意力机制就能在性能上比肩加上了注意力机制的RNN模型。
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