"""
Organize external midi dataset into a format that is compatible with the rest of the project. 
Make them usable & useful for the downstream tasks. 
"""

import sys

import step2common
import pretty_midi
import random
import params as p
import project_specific_util as u
import math

mp.RAW_MIDI_DATA_FOLDER



# ASAP_MIDI_DATASET_PATH = p.RAW_MIDI_DATA_FOLDER + "real-and-good-classical-compositions/copied-from-external-dataset/asap-dataset-with-non-performance-midi-files-removed-and-midi-files-renamed-to-contain-composer-and-composition-names/"
# u.assert_a_dir_exists(ASAP_MIDI_DATASET_PATH)
#
# # 1: delete "midi_score.mid" from "asap" dataset. midi_score.mid is NOT a performance midi. it is a score-based midi!
# asap_midi_files = u.get_all_absolute_file_paths_recursively_from_directory(ASAP_MIDI_DATASET_PATH, [".mid", ".midi"])
# for asap_midi_file in asap_midi_files:
#     if asap_midi_file.endswith("midi_score.mid"):
#         u.delete_a_file(asap_midi_file, confirmation_needed=False)
#         print(f"Deleted {asap_midi_file}")
#
# # 2: format all the midi names using step2common.py
# asap_midi_files = u.get_all_absolute_file_paths_recursively_from_directory(ASAP_MIDI_DATASET_PATH, [".mid", ".midi"])
# step2common.format_midi_file_names(asap_midi_files)
#
# # 3: remove pedals in all midi files in asap dataset and represent sustain pedal as longer duration of note
# asap_midi_files = u.get_all_absolute_file_paths_recursively_from_directory(ASAP_MIDI_DATASET_PATH, [".mid", ".midi"])
# for asap_midi_file in asap_midi_files:
#
#
#
#
# # 4: rename asap dataset midi files so that all midi file names include composer and composition names
# asap_midi_files = u.get_all_absolute_file_paths_recursively_from_directory(ASAP_MIDI_DATASET_PATH, [".mid", ".midi"])



