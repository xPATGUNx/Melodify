import py_midicsv as pm
import glob


# TODO: refactor to take MIDI file path as parameter.
def generate_csv_from_single_midi_file():
    csv_string = pm.midi_to_csv('../Midi Files/example.mid')

    with open('../CSV from MIDI/example_converted.csv', 'w') as f:
        f.writelines(csv_string)

    midi_object = pm.csv_to_midi(csv_string)

    with open('../Midi Files/example_converted.mid', 'wb') as output_file:
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)


def generate_csv_batch_from_midi_directory():
    filecounter = 1
    for filepath in glob.iglob('../Midi Files/*.mid'):
        print('Converting ' + filepath[14:] + ' to CSV.')
        csv_string = pm.midi_to_csv(filepath)
        with open('../CSV from MIDI/CSV from Midi ' + str(filecounter) + '.csv', 'w') as f:
            f.writelines(csv_string)
            filecounter += 1
    f.close()


if __name__ == '__main__':
    # generate_csv_from_single_midi_file()
    generate_csv_batch_from_midi_directory()

