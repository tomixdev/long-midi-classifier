"""
raw-midi-dataにあるmidi dataをすべて、generated-midi-imagesのfolderにうつす。

そのさい、raw-midi-dataの中のdirectory structureをそのまま維持するようにする。

つまり、raw-midi-dataのdirectoryをrecursiveに捜索した上、directory structureをそのまま、generated-midi-imagesに移す。

"""
import concurrent.futures
import os

import torch
import pandas as pd
import matplotlib.pyplot as plt

import project_specific_util as u
import params as p
import step2common

"""
use the directory name one below this directory as labels
for example:
    raw-midi-data-dir
    ├── compositions-type-A
        ├── dir1
        ├── dir2
        ├── dir3
    ├── compositions-type-B
        ├── dir1
        ├── dir2
        ├── dir3
    ├── compositions-type-C
    

In this example, use "composition-type-A" as label for all midi files under composiiton-type-A。
つまりRAW_MIDI_DATA_FOLDERの一つ以下の階層のみをラベルとして使い、それより下のsub directoriesは、すべて無視して、一箇所にまとめる。
"""


class Midi2ImgFuncs:
    """
    横長imageをパソコンで開けないことに関して。
    Note:
        - it seems that images whose width is larger than around 100000 do not open on photo viewer or editor software.
        - HOWEVER, I can still load such a big image into torch tensor in python.
        - SO, images with extremely long width should not be a big problem....多分問題なし。。。。
    """

    @classmethod
    def bw_piano_roll_img_with_pitch_as_row(cls, midi_df, midi_img_path):
        assert isinstance(midi_df, pd.DataFrame), "midi_df must be a pandas DataFrame but is {}".format(type(midi_df))
        assert isinstance(midi_img_path, str), "midi_img_path must be a string but is {}".format(type(midi_img_path))
        assert midi_img_path.endswith(".png") or midi_img_path.endswith(".jpeg"), midi_img_path


        n_row = 128
        n_col = (midi_df.end_col_num.max() - midi_df.start_col_num.min() + 1).astype(int)

        midi_img_tensor = torch.zeros(n_row, n_col)

        # fill in the image
        for _, a_midi_df_row in midi_df.iterrows():
            img_row_num = n_row - int(a_midi_df_row.pitch) # TODO: I might need to subtract one? If midi value starts from one, I need to subtract 1. If the midi value starts from zero, I do not need to do anything.
            img_col_num_start = int(a_midi_df_row.start_col_num)
            img_col_num_end = int(a_midi_df_row.end_col_num)

            # scale value to 0 and 1
            pixel_val = a_midi_df_row.velocity / 127.0
            midi_img_tensor[img_row_num, img_col_num_start:img_col_num_end] = pixel_val

        plt.imsave(midi_img_path, arr=midi_img_tensor.numpy(), vmin=0., vmax=1., cmap="gray", format='png')
        print(f"\n Generated {midi_img_path} \n")

    @classmethod
    def bw_piano_roll_img_with_velocity_as_row(cls, midi_df, midi_img_path):
        assert isinstance(midi_df, pd.DataFrame), "midi_df must be a pandas DataFrame but is {}".format(type(midi_df))
        assert isinstance(midi_img_path, str), "midi_img_path must be a string but is {}".format(type(midi_img_path))
        assert midi_img_path.endswith(".png") or midi_img_path.endswith(".jpeg"), midi_img_path

        n_row = 128
        n_col = (midi_df.end_col_num.max() - midi_df.start_col_num.min() + 1).astype(int)

        midi_img_tensor = torch.zeros(n_row, n_col)

        # fill in the image
        for _, a_midi_df_row in midi_df.iterrows():
            img_row_num = n_row - int(a_midi_df_row.velocity)  # TODO: I might need to subtract one? If midi value starts from one, I need to subtract 1. If the midi value starts from zero, I do not need to do anything.
            img_col_num_start = int(a_midi_df_row.start_col_num)
            img_col_num_end = int(a_midi_df_row.end_col_num)

            # scale value to 0 and 1
            pixel_val = a_midi_df_row.pitch / 127.0
            midi_img_tensor[img_row_num, img_col_num_start:img_col_num_end] = pixel_val

        plt.imsave(midi_img_path, arr=midi_img_tensor.numpy(), vmin=0., vmax=1., cmap="gray", format='png')
        print(f"\n Generated {midi_img_path} \n")

    # TODO: Implement four functions below
    # def sparse_color_img_with_pitch_as_row(self):
    #     raise NotImplementedError
    #
    # def sparse_color_img_with_velocity_as_row (cls, midi_file_path, midi_img_path):
    #     raise NotImplementedError
    #
    # def sparse_color_img_with_duration_as_row (cls, midi_file_path, midi_img_path):
    #     raise NotImplementedError
    #
    # def sparse_color_img_with_program_num_as_row(cls, midi_file_path, midi_img_path):
    #     raise NotImplementedError


if __name__ == "__main__":
    # Format Midi File Names -------------------------------------------------------------------------------------------
    midi_file_list = u.get_all_absolute_file_paths_recursively_from_directory(
            p.RAW_MIDI_DATA_FOLDER, file_extension_list=[".mid", ".midi"]
    )

    step2common.format_midi_file_names(midi_file_list)

    # get all midi files from RAW_MIDI_DATA_FOLDER ---------------------------------------------------------------------
    midi_file_list = u.get_all_absolute_file_paths_recursively_from_directory(
            p.RAW_MIDI_DATA_FOLDER, file_extension_list=[".mid", ".midi"]
    )

    # get directory names under RAW_MIDI_DATA_FOLDER
    dir_names_as_labels = [d for d in os.listdir(p.RAW_MIDI_DATA_FOLDER) if
                           os.path.isdir(os.path.join(p.RAW_MIDI_DATA_FOLDER, d))]

    midi2img_funcs = [getattr(Midi2ImgFuncs, a_method_name) for a_method_name in dir(Midi2ImgFuncs)
                      if callable(getattr(Midi2ImgFuncs, a_method_name)) and not a_method_name.startswith("__")]


    futures = []

    for midi2img_func in midi2img_funcs:
        save_img_root = p.GENERATED_MIDI_IMAGES_FOLDER + midi2img_func.__name__ + "/"

        # Create these directories in GENERATED_MIDI_IMAGES_FOLDER if not exists
        for dir_name in dir_names_as_labels:
            dir_path = os.path.join(save_img_root, dir_name)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            # TODO: Very inefficient. I need to find a better way to do this triple for-loop structure-
            for a_midi_file_path in midi_file_list:
                midi_df = step2common.get_midi_df(a_midi_file_path)

                for midi2img_func in midi2img_funcs:
                    save_img_root = p.GENERATED_MIDI_IMAGES_FOLDER + midi2img_func.__name__ + "/"

                    # convert RAW_MIDI_DATA_FOLDER to absolute path
                    target_img_file_name = a_midi_file_path.rsplit(sep='/')[-1].replace(".mid", ".png")

                    # TODO: This feels like an extremely insufficient and foolish way of doing if there are a lot of labels. But for now, I keep it foolish.
                    # get image label
                    target_img_label = None
                    l = a_midi_file_path.rsplit('/')
                    for a_label in dir_names_as_labels:
                        if a_label in l:
                            target_img_label = a_label
                            break

                    img_path = save_img_root + target_img_label + "/" + target_img_file_name

                    if os.path.isfile(img_path):
                        continue
                    else:
                        # midi2img_func(midi_df, img_path)
                        futures.append(executor.submit(midi2img_func, midi_df, img_path))

            concurrent.futures.wait(futures)


                # # get id=... from the file name
                # id_str = img_path.split("/")[-1].split("(")[-1].split(")")[0]
                #
                # # check if there is a file that contains the same id in the GENERATED_MIDI_IMAGES_FOLDER + config_str folder
                # all_png_images = u.get_all_absolute_file_paths_recursively_from_directory(
                #     save_img_root, file_extension_list=[".png"]
                # )
                #
                # is_midi_img_already_generated = False
                # for a_png_image_path in all_png_images:
                #     if id_str in a_png_image_path:
                #         is_midi_img_already_generated = True
                #         break
                #
                # if is_midi_img_already_generated:
                #     continue
                # else:
                #     midi2img_func(midi_df, img_path)

