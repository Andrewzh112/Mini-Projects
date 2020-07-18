import music21 as M
import torch
import config
import argparse



def save_melody(melody, file_name, step_duration=0.25, format='midi'):

    stream = M.stream.Stream()

    start_symbol = None # dummy head
    step_count = 1

    for i, symbol in enumerate(melody):
        if symbol != '_' or i == len(melody)-1:
            if start_symbol is not None:
                duration = step_duration*step_count

                if start_symbol == 'r':
                    event = M.note.Rest(quarterLength=duration)
                else:
                    event = M.note.Note(int(start_symbol), quarterLength=duration)
                stream.append(event)

                step_count = 1 # reset

            start_symbol = symbol

        else: # prolongation
            step_count += 1
    stream.write(format, file_name)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generating melodies!')
    parser.add_argument('file_name', type=str, 
                    help='output file name for melody')
    parser.add_argument('--seed', type=str, default=config.SEED,
                    help='default: "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"')
    parser.add_argument('--temperature', type=float, default=config.TEMPERATURE,
                        help='degree of exploration, higher the more random')
    args = parser.parse_args()

    model = torch.load(config.MODEL_PATH)
    melody = model.generate_melody(config.NOTES_MAPPING, args.seed, 
                                config.NUM_STEPS, args.temperature)
    # melody_long = model.generate_melody(config.NOTES_MAPPING, ' '.join(melody[:64]), 
    #                             config.NUM_STEPS, config.TEMPERATURE)
    save_melody(melody, f'{args.file_name}.mid')