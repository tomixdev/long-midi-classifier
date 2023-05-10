import sys
import pandas as pd
import project_specific_util as u
import time
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import torch
import pprint
import re
import os
import params as p
import traceback

def get_midi_df(midi_file_path):
    assert isinstance(midi_file_path, str), f"{midi_file_path = } is not a string..."
    u.assert_file_exists(midi_file_path)

    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # get grid length ------------------------------------------------------------
    tempo = None
    if midi_data.get_tempo_changes()[1].size == 1:
        tempo = midi_data.get_tempo_changes()[1]
    else:
        raise NotImplementedError(
            f"The midi has two changes multiple tempi. This situation is not assumed! Error in the following midi file: \n {midi_file_path = }")

    grid_width_in_s = (60.0 / tempo / midi_data.resolution).item()

    # make pandas dataframe -------------------------------------------------------
    midi_df = pd.DataFrame({})

    for instrument in midi_data.instruments:
        # flatten all instruments
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            duration = end - start

            program_num = instrument.program

            # round to the nearest integer number
            start_col_num = int(np.round(start / grid_width_in_s).item())  # TODO: I might need to add or subtract 1
            end_col_num = int(np.round(end / grid_width_in_s).item())  # TODO: I might need to add or subtract 1

            new_row = pd.DataFrame({
                "start_col_num": [start_col_num],  # midi note starts at the beginning of this grid
                "start_time_in_s": [start],
                "end_col_num": [end_col_num],  # midi note starts at the END of this grid
                "end_time_in_s": [end],
                "program_num": [program_num],
                "pitch": [pitch],
                "velocity": [velocity],
                "duration_in_s": [duration]
            })

            # append an element to dataframe
            midi_df = pd.concat([midi_df, new_row], ignore_index=True)

    try:
        # Cut the beginning head ----------------------------------------------------------
        min_col_num = midi_df["start_col_num"].min()
        midi_df["start_col_num"] = midi_df["start_col_num"] - min_col_num
        midi_df["end_col_num"] = midi_df["end_col_num"] - min_col_num

        min_start_time_in_s = midi_df["start_time_in_s"].min()
        midi_df["start_time_in_s"] = midi_df["start_time_in_s"] - min_start_time_in_s
        midi_df["end_time_in_s"] = midi_df["end_time_in_s"] - min_start_time_in_s
    except KeyError as e:
        print ("\n")
        print (f"Error occured for {midi_file_path} !!!!!!!!!!!")
        print ("\n")
        traceback.print_exc()
        sys.exit(1)

    return midi_df


def format_midi_file_names (midi_file_full_path_list):
    # add an ID to each midi file by renaming each file with its hash value
    # replace empty space with '-', replace '_' with '-'
    # FILE FORMAT: "this-is-file-name(id=a47baf4dce81c601775d58f23a0602c51889b79d).mid"
    for current_midi_full_file_path in midi_file_full_path_list:
        u.assert_file_exists(current_midi_full_file_path)

        # compute hash value of the file
        id_str = "id=" + u.hash_midi(current_midi_full_file_path)

        new_file_name = current_midi_full_file_path.split("/")[-1].replace("_", "-")
        new_file_name = re.sub(r"\s+", "-", new_file_name)  # replace empty space with '-'
        new_file_name = new_file_name.replace(".mid", "")
        new_file_name = new_file_name.replace(".midi", "")
        new_file_name = new_file_name.replace(",", "")
        new_file_name = new_file_name.replace(".", "")

        if id_str in current_midi_full_file_path and new_file_name in current_midi_full_file_path:
            continue

        if "id=" in current_midi_full_file_path and id_str not in current_midi_full_file_path:
            new_file_name = new_file_name.split("(", 1)[0]
            new_midi_file_full_path = current_midi_full_file_path.rsplit("/", 1)[0] + "/" + new_file_name + "(" + id_str + ")" + ".mid"
        elif "id=" in current_midi_full_file_path and id_str in current_midi_full_file_path:
            new_midi_file_full_path = current_midi_full_file_path.rsplit("/", 1)[0] + "/" + new_file_name + ".mid"
            print (new_midi_file_full_path)
            sys.exit()
        elif "id=" not in current_midi_full_file_path: # replace file name
            new_midi_file_full_path = current_midi_full_file_path.rsplit("/", 1)[0] + "/" + new_file_name + "(" + id_str + ")" + ".mid"
        else:
            raise NotImplementedError()

        # rename the file with its hash value
        os.rename (current_midi_full_file_path, new_midi_file_full_path)