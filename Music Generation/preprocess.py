import music21 as M
import os
import json
from tqdm import tqdm
import numpy as np
import tensorflow.keras as keras

CLASSICAL_DATA_PATH = 'German'
SAVE_DIR = 'save_data'
COLLATED_PATH = 'file_dataset'
MAPPING_PATH = 'mapping.json'
SEQUENCE_LENGTH = 64
START_TOKEN, END_TOKEN = '<START>', '<END>'
VALID_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.,
    3.,
    4.
]


def load_songs(data_path):
    """load all songs from path"""
    songs = []

    for path, _, files in os.walk(data_path):
        for file in files:
            if '.krn' not in file:
                continue
            song = M.converter.parse(os.path.join(path,file))
            songs.append(song)

    return songs


def filter_song(song, valid_durations):
    """bool for filtering songs with invalid durations"""
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in valid_durations:
            return True
    return False


def transpose(song):
    """transpose song to CMaj or AMin key"""
    key = song.analyze('key')
    
    if key.mode == 'major':
        interval = M.interval.Interval(key.tonic, M.pitch.Pitch('C'))
    else:
        interval = M.interval.Interval(key.tonic, M.pitch.Pitch('A'))

    return song.transpose(interval)


def encode(song, time_step=0.25):
    """encode song with symbols"""
    encoded_song = []
    for note in song.flat.notesAndRests:
        if isinstance(note, M.note.Note):
            symbol = note.pitch.midi
        else:
            symbol = 'r'
        
        steps = int(note.duration.quarterLength / time_step)
        encoded_note = [str(symbol) if step == 0 else '_' for step in range(steps)]
        encoded_song.extend(encoded_note)

    return ' '.join(encoded_song)


def preprocess(data_path, save_dir, valid_durations):
    """process songs from given path"""
    songs = load_songs(data_path)

    for i, song in tqdm(enumerate(songs)):
        # only process if song has valid durations
        if filter_song(song, valid_durations):
            continue
        
        # transpose songs to cmaj/amin
        song = transpose(song)

        encoded_song = encode(song)

        save_path = os.path.join(save_dir, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load_song(file_path):
    """load song from txt file"""
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def collate_songs(data_path, collated_path, seq_length):
    """collate songs to one file"""
    deliminator = '/ ' * seq_length
    songs = ''

    for path, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load_song(file_path)
            songs += song + ' ' + deliminator
    
    songs = songs.rstrip()

    with open(collated_path, "w") as fp:
        fp.write(songs)

    return songs


def note2idx(songs, mapping_path):
    """vocab dictionary and save to json"""

    vocab = set(songs.split())
    mapping = {symbol:i for i, symbol in enumerate(vocab)}
    
    with open(mapping_path, "w") as fp:
        json.dump(mapping, fp, indent=4)


def songs2int(songs):
    """map symbols to int keys"""
    int_songs = []

    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    
    songs = songs.split()

    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs


def generate_sequences(sequence_length):
    """create inputs and targets data sequences"""
    inputs, targets = [], []
    songs = load_song(COLLATED_PATH)
    int_songs = songs2int(songs)

    gen_sequences = len(int_songs) - sequence_length
    for i in range(gen_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    return np.array(inputs), np.array(targets)


def run():
    preprocess(CLASSICAL_DATA_PATH, SAVE_DIR, VALID_DURATIONS)
    songs = collate_songs(SAVE_DIR, COLLATED_PATH, SEQUENCE_LENGTH)
    note2idx(songs, MAPPING_PATH)
    # inputs, targets = generate_sequences(SEQUENCE_LENGTH)


if __name__=='__main__':
    run()