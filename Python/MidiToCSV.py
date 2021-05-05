import py_midicsv as pm


def generate_csv_from_midi():
    csv_string = pm.midi_to_csv('../Midi Files/example.mid')

    with open('../CSV from MIDI/example_converted.csv', 'w') as f:
        f.writelines(csv_string)

    midi_object = pm.csv_to_midi(csv_string)

    with open('../Midi Files/example_converted.mid', 'wb') as output_file:
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)


if __name__ == '__main__':
    generate_csv_from_midi()
