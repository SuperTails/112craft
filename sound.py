import soundfile as sf
import simpleaudio as sa
import numpy

class Sound:
    def __init__(self, path):
        data, self.samplerate = sf.read(path)
        data *= 32767 / max(abs(data))
        self.data = data.astype(numpy.int16)
    
    def play(self):
        sa.play_buffer(self.data, 1, 2, self.samplerate)
