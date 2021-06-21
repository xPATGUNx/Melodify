import py_midicsv as pm
import glob


# TODO: Refactor path handling
def generate_csv_from_single_midi_file(path):
    csv_string = pm.midi_to_csv(path)

    with open('../data/Midi out.csv', 'w') as f:
        f.writelines(csv_string)

    f.close()


def generate_midi_from_single_csv_file(path):
    midi_object = pm.csv_to_midi(path)

    with open('../data/converted_demo.mid', 'wb') as output_file:
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)


def generate_csv_batch_from_midi_directory():
    file_counter = 1
    for filepath in glob.iglob('../Midi Files/*.mid'):
        print('Converting ' + filepath[14:] + ' to CSV.')
        csv_string = pm.midi_to_csv(filepath)
        with open('../CSV from MIDI/CSV from Midi ' + str(file_counter) + '.csv', 'w') as f:
            f.writelines(csv_string)
            file_counter += 1
    f.close()


def generate_consolidated_csv_from_midi_batch():

    for file in glob.iglob('../CSV from MIDI/*.csv'):
        with open(file, "r") as f:
            rows = f.readlines()[5:]
            rows = (rows[:-2])
        print('Writing: ' + file)
        with open("../data/TrainData.csv", "a") as myfile:
            for element in rows:
                myfile.write(element)


if __name__ == '__main__':
    # generate_csv_from_single_midi_file('../data/Midi Out.mid')
    # generate_csv_batch_from_midi_directory()
    generate_midi_from_single_csv_file('../data/newTest.csv')
    # generate_consolidated_csv_from_midi_batch()
