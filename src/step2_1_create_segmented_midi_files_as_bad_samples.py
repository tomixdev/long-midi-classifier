import sys

import step2common
import pretty_midi
import random
import params as p
import project_specific_util as u
import math
import os
import glob
import concurrent.futures

random.seed(p.RAND_SEED)
SEGMENTED_MIDI_DIR = p.RAW_MIDI_DATA_FOLDER + "fake-or-bad-classical-compositions/segments-from-real-compositions/"
GOOD_MIDI_DIR = p.RAW_MIDI_DATA_FOLDER + "real-and-good-classical-compositions/"
u.assert_a_dir_exists(SEGMENTED_MIDI_DIR)
u.assert_a_dir_exists(GOOD_MIDI_DIR)

# get all midi files from params
paths_to_all_good_midi_files = u.get_all_absolute_file_paths_recursively_from_directory(GOOD_MIDI_DIR, [".mid", ".midi"])
step2common.format_midi_file_names(paths_to_all_good_midi_files)
paths_to_all_good_midi_files = u.get_all_absolute_file_paths_recursively_from_directory(GOOD_MIDI_DIR, [".mid", ".midi"])

def segment_midi(midi_file_path):
    mid = pretty_midi.PrettyMIDI(midi_file_path)

    notes = []
    for instrument in mid.instruments:
        notes.extend(instrument.notes)

    start = None
    end = None
    for note in notes:
        if start is None or note.start < start:
            start = note.start
        if end is None or note.end > end:
            end = note.end

    original_midi_length_in_s = end - start



    # original_midi_length_in_s = mid.get_end_time()

    segmented_midi_length_in_s = None

    """
    TODO:
    Determine Segmented Midi Length: 以下が自分がもともとやってみた、ある程度の秒数にSegments を収める方法。こちらのほうが良いかもしれない

    ---------------
    # the following numbers are set purely because of my instituiton as a composer:
    # 60 seconds would be too short to make a meaningful musical form in general.
    # 30 seconds would be enough to make music apparently good (but formally not)
    SEGMENTED_MIDI_MIN_DUR_IN_S = 30
    SEGMENTED_MIDI_MAX_DUR_IN_S = 60

    assert SEGMENTED_MIDI_MIN_DUR_IN_S < SEGMENTED_MIDI_MAX_DUR_IN_S


    if original_midi_length_in_s <= SEGMENTED_MIDI_MIN_DUR_IN_S:
        raise ValueError(f"too short midi file: {midi_file_path}")
    elif SEGMENTED_MIDI_MIN_DUR_IN_S < original_midi_length_in_s <= SEGMENTED_MIDI_MAX_DUR_IN_S:
        # for exteremly short original midi files
        segmented_midi_length_in_s = SEGMENTED_MIDI_MIN_DUR_IN_S
    elif SEGMENTED_MIDI_MAX_DUR_IN_S < original_midi_length_in_s <= SEGMENTED_MIDI_MAX_DUR_IN_S + 15:
        # for midi files whose length just exeeds the maximum midi segment duration but not significantly
        assert (SEGMENTED_MIDI_MIN_DUR_IN_S + 15) <= SEGMENTED_MIDI_MAX_DUR_IN_S
        segmented_midi_length_in_s = random.randint (SEGMENTED_MIDI_MIN_DUR_IN_S, SEGMENTED_MIDI_MAX_DUR_IN_S - 15)
    else:
        # Most of the midi files fall into this way of determining random segment midi length
        segmented_midi_length_in_s = random.randint (SEGMENTED_MIDI_MIN_DUR_IN_S, SEGMENTED_MIDI_MAX_DUR_IN_S)
    """

    """
    TODO: Determine Segmented Midi Length: 以下が、現在のImplementation。これでいいかな？？？ Very subjectivbe Decision...
    """

    if 0 < original_midi_length_in_s < 90:
        segmented_midi_length_in_s = original_midi_length_in_s / 1.5
    elif 90 <= original_midi_length_in_s < 120:
        segmented_midi_length_in_s = 60
    elif 120 <= original_midi_length_in_s < 3840:
        segmented_midi_length_in_s = original_midi_length_in_s / 2.0
        # segmented_midi_length_in_s = random.randint(60, 90)
    #      segmented_midi_length_in_s = original_midi_length_in_s / 3
    # elif 240 <= original_midi_length_in_s < 480:  # 480 sec = 8 minutes
    #      segmented_midi_length_in_s = original_midi_length_in_s / 4
    # elif 480 <= original_midi_length_in_s < 960:  # 960 sec = 16 minutes
    #      segmented_midi_length_in_s = original_midi_length_in_s / 5
    # elif 960 <= original_midi_length_in_s < 1920:  # 1920 sec = 32 minutes
    #     segmented_midi_length_in_s = original_midi_length_in_s / 12
    # elif 1920 <= original_midi_length_in_s < 3840:  # 3840 sec = 1 hour 4 minutes
    #     segmented_midi_length_in_s = original_midi_length_in_s / 16
    else:
        raise Exception(f"too long midi file: {midi_file_path}")

    segmented_midi_length_in_s -= 0.001

    assert segmented_midi_length_in_s is not None

    segment_from_start_or_end = random.choice(["start", "end"])

    list_of_midi_section_time_tuples = []
    if segment_from_start_or_end == "start":
        segment_start_point_s = start
        segment_end_point_s = start + segmented_midi_length_in_s
        while segment_end_point_s < original_midi_length_in_s:
            time_tuple = (segment_start_point_s, segment_end_point_s)
            list_of_midi_section_time_tuples.append(time_tuple)
            segment_start_point_s += segmented_midi_length_in_s
            segment_end_point_s += segmented_midi_length_in_s
    elif segment_from_start_or_end == "end":
        segment_start_point_s = original_midi_length_in_s - segmented_midi_length_in_s
        segment_end_point_s = end
        while segment_start_point_s >= start:
            time_tuple = (segment_start_point_s, segment_end_point_s)
            list_of_midi_section_time_tuples.append(time_tuple)
            segment_start_point_s -= segmented_midi_length_in_s
            segment_end_point_s -= segmented_midi_length_in_s
    else:
        raise ValueError("segment_from_start_or_end must be either start or end")

    # pick two random time points
    random_midi_section_time_tuple = random.choice(list_of_midi_section_time_tuples)

    # get the midi section
    entire_midi_df = step2common.get_midi_df(midi_file_path)
    segment_midi_df = entire_midi_df[(entire_midi_df.start_time_in_s >= random_midi_section_time_tuple[0]) & (
                entire_midi_df.end_time_in_s <= random_midi_section_time_tuple[1])]

    # iterate over all the rows in segment_midi_df and create a new midi file
    segmented_midi_data = pretty_midi.PrettyMIDI(resolution=mid.resolution)
    instrument = pretty_midi.Instrument(program=0)

    for index, row in segment_midi_df.iterrows():
        note = pretty_midi.Note(
            velocity=int(row.velocity),
            pitch=int(row.pitch),
            start=row.start_time_in_s - random_midi_section_time_tuple[0],
            end=row.end_time_in_s - random_midi_section_time_tuple[0]
        )
        instrument.notes.append(note)

    segmented_midi_data.instruments.append(instrument)

    # write out midi
    original_midi_file_name = midi_file_path.split(sep='/')[-1]
    original_midi_file_name = original_midi_file_name.replace(".midi", "")
    original_midi_file_name = original_midi_file_name.replace(".mid", "")

    segmented_midi_path = SEGMENTED_MIDI_DIR + original_midi_file_name + \
                          f"-random-{int(round(segmented_midi_length_in_s))}-sec-segment.mid"
    segmented_midi_data.write(segmented_midi_path)

    print(f"\n generated {segmented_midi_path} \n")


if __name__ == "__main__":
    # remove all files in SEGMENTED_MIDI_DIR
    # Get a list of all files in the folder
    file_list = os.listdir(SEGMENTED_MIDI_DIR)
    # Iterate over all files in the folder and delete them

    answer = u.ask_for_input_on_terminal_and_get_true_or_false(f"Deleting all files in {SEGMENTED_MIDI_DIR} ?")
    if answer:
        for file_name in file_list:
            file_path = os.path.join(SEGMENTED_MIDI_DIR, file_name)
            if file_path.endswith(".mid") or file_path.endswith(".midi"):
                os.remove(file_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(segment_midi, filename) for filename in paths_to_all_good_midi_files]
        concurrent.futures.wait(futures)