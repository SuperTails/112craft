import soundfile as sf
import simpleaudio as sa
import numpy
import fsb5

class Sound:
    def __init__(self, path: str):
        if path.endswith('.ogg'):
            data, self.samplerate = sf.read(path)
            data *= 32767 / max(abs(data))
            self.data = data.astype(numpy.int16)
        elif path.endswith('.fsb'):
            with open(path, "rb") as f:
                data = fsb5.FSB5(f.read())

            print(data.get_sample_extension())
            for sample in data.samples:
                print(f'frq: {sample.frequency}')
                print(f'ch: {sample.channels}')
                print(f'sa: {sample.samples}')
        
            rebuilt = data.rebuild_sample(data.samples[0])
            print(rebuilt)
        else:
            raise Exception(f"Unsupported file type {path}")
    
    def play(self, halfPitch=False, volume=1.0):
        rate = self.samplerate // 2 if halfPitch else self.samplerate
        if volume != 1.0:
            data = (self.data * volume).astype(numpy.int16)
        else:
            data = self.data
        sa.play_buffer(data, 1, 2, rate)
