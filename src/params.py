import os
import project_specific_util as u

RAND_SEED = 9999


RAW_MIDI_DATA_FOLDER = "../02-raw-midi-data/"
GENERATED_MIDI_IMAGES_FOLDER = "../03-generated-midi-image-set01/"

# Parameter Assertions ---------------------------------------------------------------------------------------------
u.assert_a_dir_exists(RAW_MIDI_DATA_FOLDER)
u.assert_a_dir_exists(GENERATED_MIDI_IMAGES_FOLDER)
RAW_MIDI_DATA_FOLDER = os.path.abspath(RAW_MIDI_DATA_FOLDER) + "/"
GENERATED_MIDI_IMAGES_FOLDER = os.path.abspath(GENERATED_MIDI_IMAGES_FOLDER) + "/"