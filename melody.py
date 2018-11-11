# melody generator Jacobo Lansac
from winsound import Beep as beep
import numpy as np
import pandas as pd

import winsound


class MelodyGenerator:
    def __init__(self, bpm=120):
        self.bpm = bpm  # bits per minute
        self.full_bar = int(self.bpm / 60 * 1000)
        self.root = 440
        self.rythm_context()

    def length_ms(self, length):
        return int(self.full_bar / length)

    def rythm_context(self):
        self.blanca = int(self.full_bar / 2)
        self.negra = int(self.full_bar / 4)
        self.corch = int(self.full_bar / 8)
        self.semic = int(self.full_bar / 16)

    def construct_freqs(self):
        freqs_ = [440, 466, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.91, 880]
        freqs = pd.Series(freqs_).round()
        return freqs

    def build_modo(self, grado):
        escala_base = [0, 2, 4, 5, 7, 9, 11, 12]
        escala_doble = escala_base + [12 + i for i in escala_base[1:]]
        loc = grado - 1
        _ = escala_doble[loc: loc + 8]
        escala = [i - _[0] for i in _]
        return escala

    def play_escala(self, freqs, escala):
        for idx, note in freqs.iteritems():
            if idx in escala:
                beep(int(note), self.corch)
        return None

    def rythm_generator(self):

        choices = [4,8,16]
        total = self.full_bar
        lengths = []
        time = 0
        while time < total:
            new_length = self.length_ms(np.random.choice(choices))
            if new_length <= (total - time):
                time = time + new_length
                lengths.append(new_length)

        # for ln in lengths:
        #     beep(self.root, ln)

        self.lengths = lengths
        return lengths

# ===================================================================================
# Testing escales
# ===================================================================================
mg = MelodyGenerator()
mg.rythm_generator()



freqs = mg.construct_freqs()
escala_mayor = mg.build_modo(1)
mg.play_escala(freqs, escala_mayor)
for grado in range(1, 7):
    mg.play_escala(freqs, mg.build_modo(grado))
