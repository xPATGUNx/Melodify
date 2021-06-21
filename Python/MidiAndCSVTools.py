import py_midicsv as pm
import glob


# TODO: Refactor path handling


def generate_csv_from_single_midi_file(path):
    csv_string = pm.midi_to_csv(path)

    with open('../data/Midi out.csv', 'w') as f:
        f.writelines(csv_string)

    f.close()


def generate_midi_from_single_csv_file(path_to_source, path_to_generated):
    midi_object = pm.csv_to_midi(path_to_source)

    with open(path_to_generated, 'wb') as output_file:
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


def generate_midi_from_output(csv_string, output_path):

    buffer_path = '../data/OutputBuffer.csv'

    arranged_csv_string_list = []

    with open(buffer_path, 'w') as buffer_csv:
        buffer_csv.writelines(csv_string)
    buffer_csv.close()

    with open(buffer_path, 'r') as buffer_reader:
        buffer = buffer_reader.readlines()
    buffer_reader.close()

    with open(buffer_path, 'w') as buffer_csv:
        buffer_csv.writelines(buffer[:-1])
    buffer_csv.close()

    with open('../data/Header.csv', 'r') as f:
        header = f.readlines()
    f.close()

    with open(buffer_path, 'r') as buffer_reader:
        buffer = buffer_reader.readlines()
    buffer_reader.close()

    end_line = buffer[-1]

    if end_line[5] == ',':
        end_line = end_line[:5]
    elif end_line[6] == ',':
        end_line = end_line[:6]
    else:
        end_line = end_line[:7]

    end_line = (end_line + ', End_track\n')
    end_file = '0, 0, End_of_file\n'

    arranged_csv_string_list.append(header)
    arranged_csv_string_list.append(buffer)
    arranged_csv_string_list.append(end_line)
    arranged_csv_string_list.append(end_file)

    with open('../Generated Midi/generated.csv', 'w') as f:
        for lines in arranged_csv_string_list:
            f.writelines(lines)

    generate_midi_from_single_csv_file('../Generated Midi/generated.csv', output_path)


if __name__ == '__main__':
    # generate_csv_from_single_midi_file('../data/Midi Out.mid')
    # generate_csv_batch_from_midi_directory()
    generate_midi_from_single_csv_file('../data/newTest.csv')
    # generate_consolidated_csv_from_midi_batch()
