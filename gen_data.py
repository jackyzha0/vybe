import random
import numpy as np
import data_tools.py

def get_type(type):
    pass



def gen(length=-1,sounds=[]):
    if length == -1:
        # Gen random length from 3 to 10s
        length = random.randint(3,11)
    if not sounds:
        n = np.arange(6)
        # Gen random amount of sounds from 1 to 5
        n = n[:random.randint(1,6)]
        np.random.shuffle(n)
        print(n)

    # In range of sounds, find n range(1,6) sub sounds for each sound in directory
        # e.g. sounds
    # Crawl directory and find sounds amount of wavs
    # gen range of sounds
    # create label array
    # create wav
gen()
