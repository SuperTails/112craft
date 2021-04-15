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
    
    def play(self):
        sa.play_buffer(self.data, 1, 2, self.samplerate)
