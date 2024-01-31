import audioread
import librosa

import os
import sys
import json
import time
from tqdm import tqdm
import pickle
import hashlib
import logging
import traceback
import shutil
import soundfile as sf

import torch

from gui_data.constants import *
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps
from lib_v5.vr_network.model_param_init import ModelParameters
from lib_v5 import spec_utils
from pathlib  import Path
from separate import SeperateAttributes, SeperateDemucs, SeperateMDX, SeperateVR, save_format
from typing import List


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('UVR BEGIN')

PREVIOUS_PATCH_WIN = 'UVR_Patch_1_12_23_14_54'

is_dnd_compatible = True
banner_placement = -2

def save_data(data):
    """
    Saves given data as a .pkl (pickle) file

    Paramters:
        data(dict):
            Dictionary containing all the necessary data to save
    """
    # Open data file, create it if it does not exist
    with open('data.pkl', 'wb') as data_file:
        pickle.dump(data, data_file)

def load_data() -> dict:
    """
    Loads saved pkl file and returns the stored data

    Returns(dict):
        Dictionary containing all the saved data
    """
    try:
        with open('data.pkl', 'rb') as data_file:  # Open data file
            data = pickle.load(data_file)

        return data
    except (ValueError, FileNotFoundError):
        # Data File is corrupted or not found so recreate it

        save_data(data=DEFAULT_DATA)

        return load_data()

def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''

    with open(dictionary) as d:
        data = d.read()

    return json.loads(data)

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)  # Change the current working directory to the base path

debugger = []

#--Constants--
#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_data.json')
DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')

#Style
ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.ico')
FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'centurygothic', 'GOTHIC.TTF')#ensemble_temps

#Other
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
CHANGE_LOG = os.path.join(BASE_PATH, 'gui_data', 'change_log.txt')
SPLASH_DOC = os.path.join(BASE_PATH, 'tmp', 'splash.txt')

file_check(os.path.join(MODELS_DIR, 'Main_Models'), VR_MODELS_DIR)
file_check(os.path.join(DEMUCS_MODELS_DIR, 'v3_repo'), DEMUCS_NEWER_REPO_DIR)
remove_unneeded_yamls(DEMUCS_MODELS_DIR)

remove_temps(ENSEMBLE_TEMP_PATH)
remove_temps(SAMPLE_CLIP_PATH)
remove_temps(os.path.join(BASE_PATH, 'img'))

if not os.path.isdir(ENSEMBLE_TEMP_PATH):
    os.mkdir(ENSEMBLE_TEMP_PATH)
    
if not os.path.isdir(SAMPLE_CLIP_PATH):
    os.mkdir(SAMPLE_CLIP_PATH)

model_hash_table = {}
data = load_data()
    
class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False):

        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var.get() else -1
        self.is_normalization = root.is_normalization_var.get()
        self.is_primary_stem_only = root.is_primary_stem_only_var.get()
        self.is_secondary_stem_only = root.is_secondary_stem_only_var.get()
        self.is_denoise = root.is_denoise_var.get()
        self.mdx_batch_size = 1 if root.mdx_batch_size_var.get() == DEF_OPT else int(root.mdx_batch_size_var.get())
        self.is_mdx_ckpt = False
        self.wav_type_set = root.wav_type_set
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        self.is_invert_spec = root.is_invert_spec_var.get()
        self.is_mixer_mode = root.is_mixer_mode_var.get()
        self.demucs_stems = root.demucs_stems_var.get()
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None

        if selected_process_method == ENSEMBLE_MODE:
            partitioned_name = model_name.partition(ENSEMBLE_PARTITION)
            self.process_method = partitioned_name[0]
            self.model_name = partitioned_name[2]
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = root.return_ensemble_stems()
            self.is_ensemble_mode = True if not is_secondary_model and not is_pre_proc_model else False
            self.is_4_stem_ensemble = True if root.ensemble_main_stem_var.get() == FOUR_STEM_ENSEMBLE and self.is_ensemble_mode else False
            self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if not self.ensemble_primary_stem == VOCAL_STEM else False

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = root.vr_is_secondary_model_activate_var.get() if not self.is_secondary_model else False
            self.aggression_setting = float(int(root.aggression_setting_var.get())/100)
            self.is_tta = root.is_tta_var.get()
            self.is_post_process = root.is_post_process_var.get()
            self.window_size = int(root.window_size_var.get())
            self.batch_size = 1 if root.batch_size_var.get() == DEF_OPT else int(root.batch_size_var.get())
            self.crop_size = int(root.crop_size_var.get())
            self.is_high_end_process = 'mirroring' if root.is_high_end_process_var.get() else 'None'
            self.post_process_threshold = float(root.post_process_threshold_var.get())
            self.model_capacity = 32, 128
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            self.get_model_hash()
            if self.model_hash:
                self.model_data = self.get_model_data(VR_HASH_DIR, root.vr_hash_MAPPER) if not self.model_hash == WOOD_INST_MODEL_HASH else WOOD_INST_PARAMS
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                    if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                        self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                        self.is_vr_51_model = True
                else:
                    self.model_status = False
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = root.mdx_is_secondary_model_activate_var.get() if not is_secondary_model else False
            self.margin = int(root.margin_var.get())
            self.chunks = root.determine_auto_chunks(root.chunks_var.get(), self.is_gpu_conversion) if root.is_chunk_mdxnet_var.get() else 0
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_data = self.get_model_data(MDX_HASH_DIR, root.mdx_hash_MAPPER)
                if self.model_data:
                    self.compensate = self.model_data["compensate"] if root.compensate_var.get() == AUTO_SELECT else float(root.compensate_var.get())
                    self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                    self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                    self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = root.demucs_is_secondary_model_activate_var.get() if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if not root.demucs_stems_var.get() in [VOCAL_STEM, INST_STEM] else False
            self.overlap = float(root.overlap_var.get())
            self.margin_demucs = int(root.margin_demucs_var.get())
            self.chunks_demucs = root.determine_auto_chunks(root.chunks_demucs_var.get(), self.is_gpu_conversion)
            self.shifts = int(root.shifts_var.get())
            self.is_split_mode = root.is_split_mode_var.get()
            self.segment = root.segment_var.get()
            self.is_chunk_demucs = root.is_chunk_demucs_var.get()
            self.is_demucs_combine_stems = root.is_demucs_combine_stems_var.get()
            self.is_primary_stem_only = root.is_primary_stem_only_var.get() if self.is_ensemble_mode else root.is_primary_stem_only_Demucs_var.get() 
            self.is_secondary_stem_only = root.is_secondary_stem_only_var.get() if self.is_ensemble_mode else root.is_secondary_stem_only_Demucs_var.get()
            self.get_demucs_model_path()
            self.get_demucs_model_data()

        self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0] if self.model_status else None
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        if self.is_secondary_model_activated and self.model_status:
            if (not self.is_ensemble_mode and root.demucs_stems_var.get() == ALL_STEMS and self.process_method == DEMUCS_ARCH_TYPE) or self.is_4_stem_ensemble:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = False if all(i is None for i in self.secondary_model_4_stem) else True
                self.demucs_4_stem_added_count = self.demucs_4_stem_added_count - 1 if self.is_secondary_model_activated else self.demucs_4_stem_added_count
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [None if i is None else i.model_basename for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True 
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and self.process_method == DEMUCS_ARCH_TYPE else self.primary_stem
                self.secondary_model_data(primary_stem)
                
        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model_activated = True
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.is_demucs_pre_proc_model_inst_mix = root.is_demucs_pre_proc_model_inst_mix_var.get() if self.pre_proc_model else False

    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
              
    def get_mdx_model_path(self):
        
        if self.model_name.endswith(CKPT):
            # self.chunks = 0
            # self.is_mdx_batch_mode = True
            self.is_mdx_ckpt = True
            
        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in root.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = [True for x in DEMUCS_NEWER_TAGS if x in self.model_name]
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in root.demucs_name_select_MAPPER.items():
            if self.model_name in chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        self.demucs_source_list = DEMUCS_2_SOURCE if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE
        self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE_MAPPER
        self.demucs_stem_count = 2 if DEMUCS_UVR_MODEL in self.model_name else 4
        
        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]

    def get_model_data(self, model_hash_dir, hash_mapper):
        model_settings_json = os.path.join(model_hash_dir, "{}.json".format(self.model_hash))

        if os.path.isfile(model_settings_json):
            return json.load(open(model_settings_json))
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings
            else:
                return self.get_model_data_from_popup()

    def get_model_data_from_popup(self):
        return None

    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)


class Ensembler():
    def __init__(self, is_manual_ensemble=False):
        self.is_save_all_outputs_ensemble = root.is_save_all_outputs_ensemble_var.get()
        chosen_ensemble_name = '{}'.format(root.chosen_ensemble_var.get().replace(" ", "_")) if not root.chosen_ensemble_var.get() == CHOOSE_ENSEMBLE_OPTION else 'Ensembled'
        ensemble_algorithm = root.ensemble_type_var.get().partition("/")
        ensemble_main_stem_pair = root.ensemble_main_stem_var.get().partition("/")
        time_stamp = round(time.time())
        self.audio_tool = MANUAL_ENSEMBLE
        self.main_export_path = Path(root.export_path_var.get())
        self.chosen_ensemble = f"_{chosen_ensemble_name}" if root.is_append_ensemble_name_var.get() else ''
        ensemble_folder_name = self.main_export_path if self.is_save_all_outputs_ensemble else ENSEMBLE_TEMP_PATH
        self.ensemble_folder_name = os.path.join(ensemble_folder_name, '{}_Outputs_{}'.format(chosen_ensemble_name, time_stamp))
        self.is_testing_audio = f"{time_stamp}_" if root.is_testing_audio_var.get() else ''
        self.primary_algorithm = ensemble_algorithm[0]
        self.secondary_algorithm = ensemble_algorithm[2]
        self.ensemble_primary_stem = ensemble_main_stem_pair[0]
        self.ensemble_secondary_stem = ensemble_main_stem_pair[2]
        self.is_normalization = root.is_normalization_var.get()
        self.wav_type_set = root.wav_type_set
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        if not is_manual_ensemble:
            os.mkdir(self.ensemble_folder_name)

    def ensemble_outputs(self, audio_file_base, export_path, stem, is_4_stem=False, is_inst_mix=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        if is_4_stem:
            algorithm = root.ensemble_type_var.get()
            stem_tag = stem
        else:
            if is_inst_mix:
                algorithm = self.secondary_algorithm
                stem_tag = f"{self.ensemble_secondary_stem} {INST_STEM}"
            else:
                algorithm = self.primary_algorithm if stem == PRIMARY_STEM else self.secondary_algorithm
                stem_tag = self.ensemble_primary_stem if stem == PRIMARY_STEM else self.ensemble_secondary_stem

        stem_outputs = self.get_files_to_ensemble(folder=export_path, prefix=audio_file_base, suffix=f"_({stem_tag}).wav")
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}_({stem_tag})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}.wav'.format(audio_file_output))
        
        if stem_outputs:
            spec_utils.ensemble_inputs(stem_outputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path)
            save_format(stem_save_path, self.save_format, self.mp3_bit_set)
        
        if self.is_save_all_outputs_ensemble:
            for i in stem_outputs:
                save_format(i, self.save_format, self.mp3_bit_set)
        else:
            for i in stem_outputs:
                try:
                    os.remove(i)
                except Exception as e:
                    print(e)

    def ensemble_manual(self, audio_inputs, audio_file_base, is_bulk=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        is_mv_sep = True
        
        if is_bulk:
            number_list = list(set([os.path.basename(i).split("_")[0] for i in audio_inputs]))
            for n in number_list:
                current_list = [i for i in audio_inputs if os.path.basename(i).startswith(n)]
                audio_file_base = os.path.basename(current_list[0]).split('.wav')[0]
                stem_testing = "instrum" if "Instrumental" in audio_file_base else "vocals"
                if is_mv_sep:
                    audio_file_base = audio_file_base.split("_")
                    audio_file_base = f"{audio_file_base[1]}_{audio_file_base[2]}_{stem_testing}"
                self.ensemble_manual_process(current_list, audio_file_base, is_bulk)
        else:
            self.ensemble_manual_process(audio_inputs, audio_file_base, is_bulk)
            
    def ensemble_manual_process(self, audio_inputs, audio_file_base, is_bulk):
        
        algorithm = root.choose_algorithm_var.get()
        algorithm_text = "" if is_bulk else f"_({root.choose_algorithm_var.get()})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}{}{}.wav'.format(self.is_testing_audio, audio_file_base, algorithm_text))
        spec_utils.ensemble_inputs(audio_inputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path)
        save_format(stem_save_path, self.save_format, self.mp3_bit_set)

    def get_files_to_ensemble(self, folder="", prefix="", suffix=""):
        """Grab all the files to be ensembled"""
        
        return [os.path.join(folder, i) for i in os.listdir(folder) if i.startswith(prefix) and i.endswith(suffix)]


def secondary_stem(stem):
    """Determines secondary stem"""
    
    for key, value in STEM_PAIR_MAPPER.items():
        if stem in key:
            secondary_stem = value
    
    return secondary_stem


class UVRInterface:
    def __init__(self) -> None:
        pass

    def assemble_model_data(self, model=None, arch_type=ENSEMBLE_MODE, is_dry_check=False) -> List[ModelData]:
        if arch_type == ENSEMBLE_STEM_CHECK:
            model_data = self.model_data_table
            missing_models = [model.model_status for model in model_data if not model.model_status]
            
            if missing_models or not model_data:
                model_data: List[ModelData] = [ModelData(model_name, is_dry_check=is_dry_check) for model_name in self.ensemble_model_list]
                self.model_data_table = model_data

        if arch_type == ENSEMBLE_MODE:
            model_data: List[ModelData] = [ModelData(model_name) for model_name in self.ensemble_listbox_get_all_selected_models()]
        if arch_type == ENSEMBLE_CHECK:
            model_data: List[ModelData] = [ModelData(model)]
        if arch_type == VR_ARCH_TYPE or arch_type == VR_ARCH_PM:
            model_data: List[ModelData] = [ModelData(model, VR_ARCH_TYPE)]
        if arch_type == MDX_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, MDX_ARCH_TYPE)]
        if arch_type == DEMUCS_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, DEMUCS_ARCH_TYPE)]#

        return model_data

    def create_sample(self, audio_file, sample_path=SAMPLE_CLIP_PATH):
        try:
            with audioread.audio_open(audio_file) as f:
                track_length = int(f.duration)
        except Exception as e:
            print('Audioread failed to get duration. Trying Librosa...')
            y, sr = librosa.load(audio_file, mono=False, sr=44100)
            track_length = int(librosa.get_duration(y=y, sr=sr))
        
        clip_duration = int(root.model_sample_mode_duration_var.get())
        
        if track_length >= clip_duration:
            offset_cut = track_length//3
            off_cut = offset_cut + track_length
            if not off_cut >= clip_duration:
                offset_cut = 0
            name_apped = f'{clip_duration}_second_'
        else:
            offset_cut, clip_duration = 0, track_length
            name_apped = ''

        sample = librosa.load(audio_file, offset=offset_cut, duration=clip_duration, mono=False, sr=44100)[0].T
        audio_sample = os.path.join(sample_path, f'{os.path.splitext(os.path.basename(audio_file))[0]}_{name_apped}sample.wav')
        sf.write(audio_sample, sample, 44100)
        
        return audio_sample
    
    def verify_audio(self, audio_file, is_process=True, sample_path=None):
        is_good = False
        error_data = ''
        
        if os.path.isfile(audio_file):
            try:
                librosa.load(audio_file, duration=3, mono=False, sr=44100) if not type(sample_path) is str else self.create_sample(audio_file, sample_path)
                is_good = True
            except Exception as e:
                error_name = f'{type(e).__name__}'
                traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                message = f'{error_name}: "{e}"\n{traceback_text}"'
                if is_process:
                    audio_base_name = os.path.basename(audio_file)
                    self.error_log_var.set(f'Error Loading the Following File:\n\n\"{audio_base_name}\"\n\nRaw Error Details:\n\n{message}')
                else:
                    error_data = AUDIO_VERIFICATION_CHECK(audio_file, message)

        if is_process:
            return is_good
        else:
            return is_good, error_data

    def cached_sources_clear(self):
        self.vr_cache_source_mapper = {}
        self.mdx_cache_source_mapper = {}
        self.demucs_cache_source_mapper = {}
      
    def cached_model_source_holder(self, process_method, sources, model_name=None):
        if process_method == VR_ARCH_TYPE:
            self.vr_cache_source_mapper = {**self.vr_cache_source_mapper, **{model_name: sources}}
        if process_method == MDX_ARCH_TYPE:
            self.mdx_cache_source_mapper = {**self.mdx_cache_source_mapper, **{model_name: sources}}
        if process_method == DEMUCS_ARCH_TYPE:
            self.demucs_cache_source_mapper = {**self.demucs_cache_source_mapper, **{model_name: sources}}
                             
    def cached_source_callback(self, process_method, model_name=None):
        model, sources = None, None
        
        if process_method == VR_ARCH_TYPE:
            mapper = self.vr_cache_source_mapper
        if process_method == MDX_ARCH_TYPE:
            mapper = self.mdx_cache_source_mapper
        if process_method == DEMUCS_ARCH_TYPE:
            mapper = self.demucs_cache_source_mapper
        
        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value
        
        return model, sources

    def cached_source_model_list_check(self, model_list: List[ModelData]):
        model: ModelData
        primary_model_names = lambda process_method:[model.model_basename if model.process_method == process_method else None for model in model_list]
        secondary_model_names = lambda process_method:[model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == process_method else None for model in model_list]

        self.vr_primary_model_names = primary_model_names(VR_ARCH_TYPE)
        self.mdx_primary_model_names = primary_model_names(MDX_ARCH_TYPE)
        self.demucs_primary_model_names = primary_model_names(DEMUCS_ARCH_TYPE)
        self.vr_secondary_model_names = secondary_model_names(VR_ARCH_TYPE)
        self.mdx_secondary_model_names = secondary_model_names(MDX_ARCH_TYPE)
        self.demucs_secondary_model_names = [model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == DEMUCS_ARCH_TYPE and not model.secondary_model is None else None for model in model_list]
        self.demucs_pre_proc_model_name = [model.pre_proc_model.model_basename if model.pre_proc_model else None for model in model_list]#list(dict.fromkeys())
        
        for model in model_list:
            if model.process_method == DEMUCS_ARCH_TYPE and model.is_demucs_4_stem_secondaries:
                if not model.is_4_stem_ensemble:
                    self.demucs_secondary_model_names = model.secondary_model_4_stem_model_names_list
                    break
                else:
                    for i in model.secondary_model_4_stem_model_names_list:
                        self.demucs_secondary_model_names.append(i)
        
        self.all_models = self.vr_primary_model_names + self.mdx_primary_model_names + self.demucs_primary_model_names + self.vr_secondary_model_names + self.mdx_secondary_model_names + self.demucs_secondary_model_names + self.demucs_pre_proc_model_name

    def process(self, model_name, arch_type, audio_file, export_path, is_model_sample_mode=False, is_4_stem_ensemble=False, set_progress_func=None, console_write=print) -> SeperateAttributes:
        stime = time.perf_counter()
        time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'

        if arch_type==ENSEMBLE_MODE:
            model_list, ensemble = self.assemble_model_data(), Ensembler()
            export_path = ensemble.ensemble_folder_name
            is_ensemble = True
        else:
            model_list = self.assemble_model_data(model_name, arch_type)
            is_ensemble = False
        self.cached_source_model_list_check(model_list)
        model = model_list[0]

        if self.verify_audio(audio_file):
            audio_file = self.create_sample(audio_file) if is_model_sample_mode else audio_file
        else:
            print(f'"{os.path.basename(audio_file)}\" is missing or currupted.\n')
            exit()

        audio_file_base = f"{os.path.splitext(os.path.basename(audio_file))[0]}"
        audio_file_base = audio_file_base if is_ensemble else f"{round(time.time())}_{audio_file_base}"
        audio_file_base = audio_file_base if not is_ensemble else f"{audio_file_base}_{model.model_basename}"
        if not is_ensemble:
            audio_file_base = f"{audio_file_base}_{model.model_basename}"

        if not is_ensemble:
            export_path = os.path.join(Path(export_path), model.model_basename, os.path.splitext(os.path.basename(audio_file))[0])
            if not os.path.isdir(export_path):
                os.makedirs(export_path) 

        if set_progress_func is None:
            pbar = tqdm(total=1)
            self._progress = 0
            def set_progress_func(step, inference_iterations=0):
                progress_curr = step + inference_iterations
                pbar.update(progress_curr-self._progress)
                self._progress = progress_curr

            def postprocess():
                pbar.close()
        else:
            def postprocess():
                pass

        process_data = {
            'model_data': model,
            'export_path': export_path,
            'audio_file_base': audio_file_base,
            'audio_file': audio_file,
            'set_progress_bar': set_progress_func,
            'write_to_console': lambda progress_text, base_text='': console_write(base_text + progress_text),
            'process_iteration': lambda:None,
            'cached_source_callback': self.cached_source_callback,
            'cached_model_source_holder': self.cached_model_source_holder,
            'list_all_models': self.all_models,
            'is_ensemble_master': is_ensemble,
            'is_4_stem_ensemble': is_ensemble and is_4_stem_ensemble
        }
        if model.process_method == VR_ARCH_TYPE:
            seperator = SeperateVR(model, process_data)
        if model.process_method == MDX_ARCH_TYPE:
            seperator = SeperateMDX(model, process_data)
        if model.process_method == DEMUCS_ARCH_TYPE:
            seperator = SeperateDemucs(model, process_data)

        seperator.seperate()
        postprocess()

        if is_ensemble:
            audio_file_base = audio_file_base.replace(f"_{model.model_basename}", "")
            console_write(ENSEMBLING_OUTPUTS)
            
            if is_4_stem_ensemble:
                for output_stem in DEMUCS_4_SOURCE_LIST:
                    ensemble.ensemble_outputs(audio_file_base, export_path, output_stem, is_4_stem=True)
            else:
                if not root.is_secondary_stem_only_var.get():
                    ensemble.ensemble_outputs(audio_file_base, export_path, PRIMARY_STEM)
                if not root.is_primary_stem_only_var.get():
                    ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM)
                    ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM, is_inst_mix=True)

            console_write(DONE)

        if is_model_sample_mode:
            if os.path.isfile(audio_file):
                os.remove(audio_file)

        torch.cuda.empty_cache()
        
        if is_ensemble and len(os.listdir(export_path)) == 0:
            shutil.rmtree(export_path)
        console_write(f'Process Complete, using time: {time_elapsed()}\nOutput path: {export_path}')
        self.cached_sources_clear()
        return seperator


class RootWrapper:
    def __init__(self, var) -> None:
        self.var=var
    
    def set(self, val):
        self.var=val
    
    def get(self):
        return self.var

class FakeRoot:
    def __init__(self) -> None:
        self.wav_type_set = 'PCM_16'
        self.vr_hash_MAPPER = load_model_hash_data(VR_HASH_JSON)
        self.mdx_hash_MAPPER = load_model_hash_data(MDX_HASH_JSON)
        self.mdx_name_select_MAPPER = load_model_hash_data(MDX_MODEL_NAME_SELECT)
        self.demucs_name_select_MAPPER = load_model_hash_data(DEMUCS_MODEL_NAME_SELECT)
    
    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            wrapped=RootWrapper(None)
            super().__setattr__(__name, wrapped)
            return wrapped

    def load_saved_settings(self, loaded_setting: dict, process_method=None):
        """Loads user saved application settings or resets to default"""
        
        for key, value in DEFAULT_DATA.items():
            if not key in loaded_setting.keys():
                loaded_setting = {**loaded_setting, **{key:value}}
                loaded_setting['batch_size'] = DEF_OPT
        
        is_ensemble = True if process_method == ENSEMBLE_MODE else False
        
        if not process_method or process_method == VR_ARCH_PM or is_ensemble:
            self.vr_model_var.set(loaded_setting['vr_model'])
            self.aggression_setting_var.set(loaded_setting['aggression_setting'])
            self.window_size_var.set(loaded_setting['window_size'])
            self.batch_size_var.set(loaded_setting['batch_size'])
            self.crop_size_var.set(loaded_setting['crop_size'])
            self.is_tta_var.set(loaded_setting['is_tta'])
            self.is_output_image_var.set(loaded_setting['is_output_image'])
            self.is_post_process_var.set(loaded_setting['is_post_process'])
            self.is_high_end_process_var.set(loaded_setting['is_high_end_process'])
            self.post_process_threshold_var.set(loaded_setting['post_process_threshold'])
            self.vr_voc_inst_secondary_model_var.set(loaded_setting['vr_voc_inst_secondary_model'])
            self.vr_other_secondary_model_var.set(loaded_setting['vr_other_secondary_model'])
            self.vr_bass_secondary_model_var.set(loaded_setting['vr_bass_secondary_model'])
            self.vr_drums_secondary_model_var.set(loaded_setting['vr_drums_secondary_model'])
            self.vr_is_secondary_model_activate_var.set(loaded_setting['vr_is_secondary_model_activate'])
            self.vr_voc_inst_secondary_model_scale_var.set(loaded_setting['vr_voc_inst_secondary_model_scale'])
            self.vr_other_secondary_model_scale_var.set(loaded_setting['vr_other_secondary_model_scale'])
            self.vr_bass_secondary_model_scale_var.set(loaded_setting['vr_bass_secondary_model_scale'])
            self.vr_drums_secondary_model_scale_var.set(loaded_setting['vr_drums_secondary_model_scale'])
        
        if not process_method or process_method == DEMUCS_ARCH_TYPE or is_ensemble:
            self.demucs_model_var.set(loaded_setting['demucs_model'])
            self.segment_var.set(loaded_setting['segment'])
            self.overlap_var.set(loaded_setting['overlap'])
            self.shifts_var.set(loaded_setting['shifts'])
            self.chunks_demucs_var.set(loaded_setting['chunks_demucs'])
            self.margin_demucs_var.set(loaded_setting['margin_demucs'])
            self.is_chunk_demucs_var.set(loaded_setting['is_chunk_demucs'])
            self.is_chunk_mdxnet_var.set(loaded_setting['is_chunk_mdxnet'])
            self.is_primary_stem_only_Demucs_var.set(loaded_setting['is_primary_stem_only_Demucs'])
            self.is_secondary_stem_only_Demucs_var.set(loaded_setting['is_secondary_stem_only_Demucs'])
            self.is_split_mode_var.set(loaded_setting['is_split_mode'])
            self.is_demucs_combine_stems_var.set(loaded_setting['is_demucs_combine_stems'])
            self.demucs_voc_inst_secondary_model_var.set(loaded_setting['demucs_voc_inst_secondary_model'])
            self.demucs_other_secondary_model_var.set(loaded_setting['demucs_other_secondary_model'])
            self.demucs_bass_secondary_model_var.set(loaded_setting['demucs_bass_secondary_model'])
            self.demucs_drums_secondary_model_var.set(loaded_setting['demucs_drums_secondary_model'])
            self.demucs_is_secondary_model_activate_var.set(loaded_setting['demucs_is_secondary_model_activate'])
            self.demucs_voc_inst_secondary_model_scale_var.set(loaded_setting['demucs_voc_inst_secondary_model_scale'])
            self.demucs_other_secondary_model_scale_var.set(loaded_setting['demucs_other_secondary_model_scale'])
            self.demucs_bass_secondary_model_scale_var.set(loaded_setting['demucs_bass_secondary_model_scale'])
            self.demucs_drums_secondary_model_scale_var.set(loaded_setting['demucs_drums_secondary_model_scale'])
            self.demucs_stems_var.set(loaded_setting['demucs_stems'])
            # self.update_stem_checkbox_labels(self.demucs_stems_var.get(), demucs=True)
            self.demucs_pre_proc_model_var.set(data['demucs_pre_proc_model'])
            self.is_demucs_pre_proc_model_activate_var.set(data['is_demucs_pre_proc_model_activate'])
            self.is_demucs_pre_proc_model_inst_mix_var.set(data['is_demucs_pre_proc_model_inst_mix'])
        
        if not process_method or process_method == MDX_ARCH_TYPE or is_ensemble:
            self.mdx_net_model_var.set(loaded_setting['mdx_net_model'])
            self.chunks_var.set(loaded_setting['chunks'])
            self.margin_var.set(loaded_setting['margin'])
            self.compensate_var.set(loaded_setting['compensate'])
            self.is_denoise_var.set(loaded_setting['is_denoise'])
            self.is_invert_spec_var.set(loaded_setting['is_invert_spec'])
            self.is_mixer_mode_var.set(loaded_setting['is_mixer_mode'])
            self.mdx_batch_size_var.set(loaded_setting['mdx_batch_size'])
            self.mdx_voc_inst_secondary_model_var.set(loaded_setting['mdx_voc_inst_secondary_model'])
            self.mdx_other_secondary_model_var.set(loaded_setting['mdx_other_secondary_model'])
            self.mdx_bass_secondary_model_var.set(loaded_setting['mdx_bass_secondary_model'])
            self.mdx_drums_secondary_model_var.set(loaded_setting['mdx_drums_secondary_model'])
            self.mdx_is_secondary_model_activate_var.set(loaded_setting['mdx_is_secondary_model_activate'])
            self.mdx_voc_inst_secondary_model_scale_var.set(loaded_setting['mdx_voc_inst_secondary_model_scale'])
            self.mdx_other_secondary_model_scale_var.set(loaded_setting['mdx_other_secondary_model_scale'])
            self.mdx_bass_secondary_model_scale_var.set(loaded_setting['mdx_bass_secondary_model_scale'])
            self.mdx_drums_secondary_model_scale_var.set(loaded_setting['mdx_drums_secondary_model_scale'])
        
        if not process_method or is_ensemble:
            self.is_save_all_outputs_ensemble_var.set(loaded_setting['is_save_all_outputs_ensemble'])
            self.is_append_ensemble_name_var.set(loaded_setting['is_append_ensemble_name'])
            self.chosen_audio_tool_var.set(loaded_setting['chosen_audio_tool'])
            self.choose_algorithm_var.set(loaded_setting['choose_algorithm'])
            self.time_stretch_rate_var.set(loaded_setting['time_stretch_rate'])
            self.pitch_rate_var.set(loaded_setting['pitch_rate'])
            self.is_primary_stem_only_var.set(loaded_setting['is_primary_stem_only'])
            self.is_secondary_stem_only_var.set(loaded_setting['is_secondary_stem_only'])
            self.is_testing_audio_var.set(loaded_setting['is_testing_audio'])
            self.is_add_model_name_var.set(loaded_setting['is_add_model_name'])
            self.is_accept_any_input_var.set(loaded_setting["is_accept_any_input"])
            self.is_task_complete_var.set(loaded_setting['is_task_complete'])
            self.is_create_model_folder_var.set(loaded_setting['is_create_model_folder'])
            self.mp3_bit_set_var.set(loaded_setting['mp3_bit_set'])
            self.save_format_var.set(loaded_setting['save_format'])
            self.wav_type_set_var.set(loaded_setting['wav_type_set'])
            self.user_code_var.set(loaded_setting['user_code'])
            
        self.is_gpu_conversion_var.set(loaded_setting['is_gpu_conversion'])
        self.is_normalization_var.set(loaded_setting['is_normalization'])
        self.help_hints_var.set(loaded_setting['help_hints_var'])
        
        self.model_sample_mode_var.set(loaded_setting['model_sample_mode'])
        self.model_sample_mode_duration_var.set(loaded_setting['model_sample_mode_duration'])


root = FakeRoot()
root.load_saved_settings(DEFAULT_DATA)