#!/usr/bin/env python3

from manipulations.constants import SAMPLE_RATE, \
                                    N_FFT, \
                                    N_MELS, \
                                    HOP_LENGTH, \
                                    IMAGE_FORMAT, \
                                    SOUNDFILE_SUBTYPE

from manipulations.audio_to_spectrogram import audio_to_spectrogram
from manipulations.spectrogram_to_audio import spectrogram_to_audio

from support.launch_editor import launch_editor

from effects.image_effects import apply_blur, \
                                  apply_pixelate, \
                                  apply_waves, \
                                  apply_shuffle, \
                                  apply_threshold, \
                                  apply_twirl, \
                                  apply_skew, \
                                  apply_rotation, \
                                  apply_scale, \
                                  apply_noise, \
                                  apply_crop, \
                                  apply_offset

import gradio as gr
import numpy as np

import librosa
import soundfile

from PIL import Image

from os import makedirs
from os.path import join, realpath
from pathlib import Path
from typing import Tuple, Dict, Any

import json

class AudioInfo:
    sample_rate: int
    duration: float
    note_median: str

def on_click_convert_to_spectrogram(
    input_path: str,
    input_audio: Tuple[int, np.ndarray[Any, Any]],
    sample_rate: float,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> Tuple[
    str,
    Image.Image,
    gr.Button,
    gr.Button,
    gr.Code,
]:
    file_name_without_extension = Path(input_path).stem

    file_name_without_extension_clean = cleanup_file_name(file_name_without_extension)

    output_dir = join(
        "outputs",
        "audio-to-spectrogram",
    )

    makedirs(output_dir, exist_ok=True)

    output_name = f"{file_name_without_extension_clean}_{sample_rate}-{n_fft}-{hop_length}-{n_mels}"

    output_path = realpath(
        join(
            output_dir,
            f"{output_name}.tiff",
        )
    )

    input_path_intermediate = realpath(
        join(
            output_dir,
            f"{output_name}.wav",
        )
    )

    sample_rate_intermediate, audio_data = input_audio

    soundfile.write(
        file=input_path_intermediate,
        data=audio_data,
        samplerate=sample_rate_intermediate,
        subtype=SOUNDFILE_SUBTYPE,
    )

    audio_to_spectrogram(
        input_path_intermediate,
        output_path,
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
    )

    image = Image.open(output_path).convert("RGB")

    button_send_to_editor = gr.Button(
        value="Send to Editor",
        visible=True,
    )

    button_send_to_audio = gr.Button(
        value="Send to From Spectrogram",
        visible=True,
    )

    spectrogram_info_dict = {
        "width": image.width,
        "height": image.height,
    }

    spectrogram_info_json = json.dumps(spectrogram_info_dict, indent=2)

    code_spectrogram_info = gr.Code(
        label="Output Spectrogram Info",
        visible=True,
        value=spectrogram_info_json,
        language="json",
    )

    return output_path, \
           image, \
           button_send_to_editor, \
           button_send_to_audio, \
           code_spectrogram_info

def on_click_to_audio(
    input_path: str,
    sample_rate: float,
    n_fft: int,
    hop_length: int,
    pitch_shift: float,
    gain: float,
    gain_multiplier: float,
    clip: float,
) -> Tuple[
    str,
    gr.Audio,
    gr.Code,
]:
    file_name_without_extension = Path(input_path).stem

    output_dir = join(
        "outputs",
        "spectrogram-to-audio",
    )

    makedirs(output_dir, exist_ok=True)

    output_path = realpath(
        join(
            output_dir,
            f"{file_name_without_extension}_{sample_rate}-{n_fft}-{hop_length}-{pitch_shift}.wav",
        )
    )

    spectrogram_to_audio(
        input_path,
        output_path,
        sample_rate,
        n_fft,
        hop_length,
    )

    audio_data, sample_rate = librosa.load(output_path, sr=sample_rate)

    if abs(pitch_shift) > 0:
        audio_data = librosa.effects.pitch_shift(
            y=audio_data,
            sr=sample_rate,
            n_steps=pitch_shift,
        )

    gain_total = gain * gain_multiplier

    is_gain_changed = gain_total > 1.0 or gain_total < 1.0

    if is_gain_changed:
        audio_data = audio_data * gain_total

    if clip < 1.0 or is_gain_changed:
        audio_data = np.clip(audio_data, -clip, clip)

    soundfile.write(
        file=output_path,
        data=audio_data,
        samplerate=sample_rate,
        subtype=SOUNDFILE_SUBTYPE,
    )

    audio_output_preview = gr.Audio(
        label="Output Audio Preview",
        type="filepath",
        value=output_path,
        visible=True,
        show_download_button=True,
    )

    audio_info = get_audio_info(audio_data, sample_rate)
    audio_info_json = json.dumps(audio_info.__dict__, indent=2)

    code_audio_info = gr.Code(
        label="Input Audio Info",
        visible=True,
        value=audio_info_json,
        language="json",
    )

    return output_path, \
           audio_output_preview, \
           code_audio_info

def on_click_send_to_editor(
    image: Image.Image,
    state_tab_id: int,
) -> Tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    gr.Tabs,
]:
    tabs = gr.Tabs(selected=state_tab_id)
    return image, image, image, tabs

def on_click_send_to_audio(
    spectrogram_path: str,
    state_tab_id: int,
) -> Tuple[str, gr.Tabs]:
    tabs = gr.Tabs(selected=state_tab_id)
    return spectrogram_path, tabs

def on_click_open_in_krita(
    input_path: str,
    images: Dict[str, Image.Image]
) -> Image.Image:
    file_name_without_extension = Path(input_path).stem

    output_dir = join(
        "outputs",
        "editor",
    )

    makedirs(output_dir, exist_ok=True)

    output_path = realpath(
        join(
            output_dir,
            f"{file_name_without_extension}_krita.tiff",
        )
    )

    image = images["composite"]

    image.convert(IMAGE_FORMAT).save(output_path)

    launch_editor("krita", output_path)

    image_edited = Image.open(output_path).convert("RGB")

    return image_edited

def on_click_send_to_audio_from_editor(
    input_path: str,
    images: Dict[str, Image.Image],
    state_tab_id: int,
) -> Tuple[str, gr.Tabs]:
    file_name_without_extension = Path(input_path).stem

    output_dir = join(
        "outputs",
        "editor",
    )

    makedirs(output_dir, exist_ok=True)

    output_path = realpath(
        join(
            output_dir,
            f"{file_name_without_extension}_edited.tiff",
        )
    )

    image = images["composite"]

    image.convert(IMAGE_FORMAT).save(output_path)

    tabs = gr.Tabs(selected=state_tab_id)

    return output_path, tabs

def on_upload_input_audio(
    input_path: str,
    is_trim: bool,
    trim_threshold_db: float,
) -> Tuple[
    gr.Audio,
    int,
    int,
    gr.Code,
    gr.Button,
]:
    sample_rate = librosa.get_samplerate(input_path)

    sample_rate_int = int(sample_rate)

    audio_data, sample_rate = librosa.load(input_path, sr=sample_rate)

    if is_trim:
        audio_data, _ = librosa.effects.trim(y=audio_data, top_db=trim_threshold_db)

    audio = get_audio_preview_input(audio_data, sample_rate_int)

    audio_info = get_audio_info(audio_data, sample_rate)
    audio_info_json = json.dumps(audio_info.__dict__, indent=2)

    code_audio_info = gr.Code(
        label="Input Audio Info",
        visible=True,
        value=audio_info_json,
        language="json",
    )

    button = gr.Button(
        value="Convert",
        interactive=True,
    )

    return audio, \
           sample_rate_int, \
           sample_rate_int, \
           code_audio_info, \
           button

def on_change_trim_threshold_to_spectrogram(
    input_path: str,
    trim_threshold_db: float,
) -> Tuple[gr.Audio, str]:
    sample_rate = librosa.get_samplerate(input_path)

    sample_rate_int = int(sample_rate)

    audio_data, sample_rate = librosa.load(input_path, sr=sample_rate)

    audio_data, _ = librosa.effects.trim(y=audio_data, top_db=trim_threshold_db)

    audio_info = get_audio_info(audio_data, sample_rate)
    audio_info_json = json.dumps(audio_info.__dict__, indent=2)

    audio_preview_input = get_audio_preview_input(audio_data, sample_rate_int)

    return audio_preview_input, audio_info_json

def on_change_dropdown_effect(dropdown: gr.Dropdown, value: int) -> Tuple:
    updates = []

    for index in range(len(dropdown.choices)):
        updates.append(gr.update(visible=index == value))

    return tuple(updates)

def on_click_update_image_effects_preview(
    images: Dict[str, Image.Image],
) -> Tuple[
    Image.Image,
    Image.Image,
]:
    image = images["composite"]
    return image, image

def on_click_apply_image_effects_preview(
    image: Image.Image,
) -> Tuple[Image.Image, Image.Image]:
    return image, image

def on_change_trim_to_spectrogram(is_checked: bool) -> gr.Slider:
    if is_checked:
        return get_slider_trim_threshold_to_spectrogram()
    else:
        return gr.Slider(visible=False)

def cleanup_file_name(file_name: str) -> str:
    file_name = file_name.lower()
    file_name = file_name.replace(" ", "-")
    file_name = file_name.replace("_", "-")
    file_name = file_name.replace("(", "")
    file_name = file_name.replace(")", "")

    return file_name

def get_audio_info(
    audio_data: np.ndarray[Any, Any],
    sample_rate: float,
) -> AudioInfo:
    info = AudioInfo()

    info.sample_rate = int(sample_rate)

    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    info.duration = duration

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
    frequency_median = np.median(spectral_rolloff)
    note_median = librosa.hz_to_note(frequency_median, cents=True)
    note_median = note_median.replace("\u266f", "#")

    info.note_median = note_median

    return info

def get_audio_preview_input(
    audio_data: np.ndarray[Any, Any],
    sample_rate: int,
) -> gr.Audio:
    return gr.Audio(
        label="Input Audio Preview",
        value=(sample_rate, audio_data),
        type="numpy",
        show_download_button=False,
        visible=True,
    )

def get_slider_trim_threshold_to_spectrogram() -> gr.Slider:
    return gr.Slider(
        label="Trim Audio Threshold (dB)",
        minimum=0,
        maximum=60,
        step=0.1,
        value=60,
        visible=True,
    )

with gr.Blocks(title="spectro-noise") as blocks:
    with gr.Tabs() as tabs:
        with gr.Tab(label="To Spectrogram", id=1):
            with gr.Row():
                with gr.Column():
                    file_input_audio = gr.File(label="Input Audio File")

                    checkbox_trim_to_spectrogram = gr.Checkbox(
                        label="Trim Audio File",
                        value=True,
                    )

                    slider_trim_threshold_to_spectrogram = get_slider_trim_threshold_to_spectrogram()

                    audio_input_preview = gr.Audio(visible=False)

                    code_input_audio_info = gr.Code(visible=False)

                    slider_sample_rate_to_spectrogram = gr.Slider(
                        label="Sample Rate",
                        minimum=0,
                        maximum=192000,
                        step=1,
                        value=SAMPLE_RATE,
                    )

                    slider_n_fft_to_spectrogram = gr.Slider(
                        label="FFT Window Size",
                        minimum=0,
                        maximum=65536,
                        step=1,
                        value=N_FFT,
                    )

                    slider_hop_length_to_spectrogram = gr.Slider(
                        label="FFT Window Hop Length",
                        minimum=0,
                        maximum=65536,
                        step=1,
                        value=HOP_LENGTH,
                    )

                    slider_n_mels_to_spectrogram = gr.Slider(
                        label="Mel scale size",
                        minimum=0,
                        maximum=65536,
                        step=1,
                        value=N_MELS,
                    )

                    button_convert_to_spectrogram = gr.Button(
                        value="Convert",
                        interactive=False,
                    )

                with gr.Column():
                    button_send_to_editor = gr.Button(
                        visible=False,
                    )

                    button_send_to_audio = gr.Button(
                        visible=False,
                    )

                    textbox_output_spectrogram = gr.Textbox(
                        label="Output Spectrogram Path",
                        show_copy_button=True,
                        interactive=False,
                    )

                    code_output_spectrogram = gr.Code(visible=False)

                    image_output_spectrogram_preview = gr.Image(
                        label="Output Spectrogram Preview",
                        image_mode="RGB",
                        show_download_button=False,
                        interactive=False,
                        type="pil",
                        show_label=False,
                    )

            button_convert_to_spectrogram.click(
                fn=on_click_convert_to_spectrogram,
                inputs=[
                    file_input_audio,
                    audio_input_preview,
                    slider_sample_rate_to_spectrogram,
                    slider_n_fft_to_spectrogram,
                    slider_hop_length_to_spectrogram,
                    slider_n_mels_to_spectrogram,
                ],
                outputs=[
                    textbox_output_spectrogram,
                    image_output_spectrogram_preview,
                    button_send_to_editor,
                    button_send_to_audio,
                    code_output_spectrogram,
                ],
            )

            checkbox_trim_to_spectrogram.change(
                fn=on_change_trim_to_spectrogram,
                inputs=[
                    checkbox_trim_to_spectrogram,
                ],
                outputs={
                    slider_trim_threshold_to_spectrogram,
                },
            )

        with gr.Tab(label="Editor", id=2) as tab_editor:
            with gr.Row():
                button_send_to_audio_from_editor = gr.Button("Send to From Spectrogram")

            with gr.Row():
                image_editor_spectrogram = gr.ImageEditor(
                    label="Spectrogram Editor",
                    image_mode="RGB",
                    type="pil",
                    interactive=True,
                    sources=["clipboard", "upload"],
                    brush=gr.Brush(
                        colors=[
                            "#FFFFFF",
                            "#000000",
                        ],
                        color_mode="fixed",
                    ),
                )

            with gr.Row():
                button_open_in_krita = gr.Button("Open in Krita")

            with gr.Row():
                with gr.Column():
                    button_update_image_effects_preview = gr.Button("Update from Editor")

                    dropdown_effect = gr.Dropdown(
                        label="Effect",
                        choices=[
                            "Gaussian Blur",
                            "Pixelate",
                            "Waves",
                            "Shuffle",
                            "Threshold",
                            "Twirl",
                            "Skew",
                            "Rotation",
                            "Scale",
                            "Offset",
                            "Noise",
                            "Crop",
                        ],
                        type="index",
                    )

                    with gr.Group(visible=False) as group_effect_gaussian_blur:
                        slider_blur_radius = gr.Slider(
                            label="Radius",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.01,
                            value=0.0,
                        )

                    with gr.Group(visible=False) as group_effect_pixelate:
                        slider_pixelate_block_size = gr.Slider(
                            label="Block Size",
                            minimum=0,
                            maximum=1024,
                            step=1,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_waves:
                        slider_waves_length_horizontal = gr.Slider(
                            label="Horizontal Length",
                            minimum=0,
                            maximum=300,
                            step=1,
                            value=10,
                        )

                        slider_waves_amplitude_horizontal = gr.Slider(
                            label="Horizontal Amplitude",
                            minimum=0,
                            maximum=300,
                            step=1,
                            value=0,
                        )

                        slider_waves_shift_horizontal = gr.Slider(
                            label="Horizontal Shift",
                            minimum=0,
                            maximum=300,
                            step=1,
                            value=0,
                        )

                        slider_waves_length_vertical = gr.Slider(
                            label="Vertical Length",
                            minimum=0,
                            maximum=300,
                            step=1,
                            value=10,
                        )

                        slider_waves_amplitude_vertical = gr.Slider(
                            label="Vertical Amplitude",
                            minimum=0,
                            maximum=300,
                            step=1,
                            value=0,
                        )

                        slider_waves_shift_vertical = gr.Slider(
                            label="Vertical Shift",
                            minimum=0,
                            maximum=300,
                            step=1,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_shuffle:
                        slider_shuffle_tile_size_x = gr.Slider(
                            label="Tile Size X",
                            minimum=0,
                            maximum=1024,
                            step=1,
                            value=0,
                        )

                        slider_shuffle_tile_size_y = gr.Slider(
                            label="Tile Size Y",
                            minimum=0,
                            maximum=1024,
                            step=1,
                            value=0,
                        )

                        slider_shuffle_strength = gr.Slider(
                            label="Strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_threshold:
                        slider_threshold_value = gr.Slider(
                            label="Threshold",
                            minimum=0,
                            maximum=255,
                            step=1,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_twirl:
                        slider_twirl_angle = gr.Slider(
                            label="Angle",
                            minimum=0.0,
                            maximum=360.0,
                            step=0.1,
                            value=0,
                        )

                        slider_twirl_radius= gr.Slider(
                            label="Radius",
                            minimum=0.0,
                            maximum=1024.0,
                            step=0.1,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_skew:
                        slider_skew_angle = gr.Slider(
                            label="Angle",
                            minimum=0.0,
                            maximum=360.0,
                            step=0.1,
                            value=0,
                        )

                        slider_skew_factor= gr.Slider(
                            label="Factor",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_rotation:
                        slider_rotation_angle = gr.Slider(
                            label="Angle",
                            minimum=-360.0,
                            maximum=360.0,
                            step=0.1,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_scale:
                        slider_scale_factor_horizontal = gr.Slider(
                            label="Horizontal Factor",
                            minimum=0.01,
                            maximum=10.0,
                            step=0.01,
                            value=1,
                        )

                        slider_scale_factor_vertical = gr.Slider(
                            label="Vertical Factor",
                            minimum=0.01,
                            maximum=10.0,
                            step=0.01,
                            value=1,
                        )

                    with gr.Group(visible=False) as group_effect_offset:
                        slider_offset_x = gr.Slider(
                            label="Offset X",
                            minimum=-2048,
                            maximum=2048,
                            step=1,
                            value=0,
                        )

                        slider_offset_y = gr.Slider(
                            label="Offset Y",
                            minimum=-2048,
                            maximum=2048,
                            step=1,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_noise:
                        slider_noise_value = gr.Slider(
                            label="Factor",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0,
                        )

                    with gr.Group(visible=False) as group_effect_crop:
                        slider_crop_offset_left = gr.Slider(
                            label="Offset Left",
                            minimum=0,
                            maximum=2048,
                            step=1,
                            value=0,
                        )

                        slider_crop_offset_upper = gr.Slider(
                            label="Offset Upper",
                            minimum=0,
                            maximum=2048,
                            step=1,
                            value=0,
                        )

                        slider_crop_offset_right = gr.Slider(
                            label="Offset Right",
                            minimum=0,
                            maximum=2048,
                            step=1,
                            value=0,
                        )

                        slider_crop_offset_lower = gr.Slider(
                            label="Offset Lower",
                            minimum=0,
                            maximum=2048,
                            step=1,
                            value=0,
                        )

                    button_apply_image_effects_preview = gr.Button("Apply to Editor")

                with gr.Column():
                    image_effects_preview_spectrogram = gr.Image(
                        label="Spectrogram Effects Preview",
                        type="pil",
                        show_label=False,
                    )

                    state_effects_preview_spectrogram = gr.State()

            button_update_image_effects_preview.click(
                fn=on_click_update_image_effects_preview,
                inputs=[
                    image_editor_spectrogram,
                ],
                outputs=[
                    image_effects_preview_spectrogram,
                    state_effects_preview_spectrogram,
                ],
            )

            dropdown_effect.change(
                # HACK: Pass object of type `gr.Dropdown` as well.
                fn=lambda value: on_change_dropdown_effect(dropdown_effect, value),
                inputs=[
                    dropdown_effect,
                ],
                outputs=[
                    group_effect_gaussian_blur,
                    group_effect_pixelate,
                    group_effect_waves,
                    group_effect_shuffle,
                    group_effect_threshold,
                    group_effect_twirl,
                    group_effect_skew,
                    group_effect_rotation,
                    group_effect_scale,
                    group_effect_offset,
                    group_effect_noise,
                    group_effect_crop,
                ],
            )

            slider_blur_radius.change(
                fn=apply_blur,
                inputs=[
                    state_effects_preview_spectrogram,
                    slider_blur_radius,
                ],
                outputs=[
                    image_effects_preview_spectrogram,
                ],
                scroll_to_output=True,
            )

            slider_pixelate_block_size.change(
                fn=apply_pixelate,
                inputs=[
                    state_effects_preview_spectrogram,
                    slider_pixelate_block_size,
                ],
                outputs=[
                    image_effects_preview_spectrogram,
                ],
                scroll_to_output=True,
            )

            for slider_waves in [
                slider_waves_length_horizontal,
                slider_waves_amplitude_horizontal,
                slider_waves_shift_horizontal,
                slider_waves_length_vertical,
                slider_waves_amplitude_vertical,
                slider_waves_shift_vertical,
            ]:
                slider_waves.change(
                    fn=apply_waves,
                    inputs=[
                        state_effects_preview_spectrogram,
                        slider_waves_length_horizontal,
                        slider_waves_amplitude_horizontal,
                        slider_waves_shift_horizontal,
                        slider_waves_length_vertical,
                        slider_waves_amplitude_vertical,
                        slider_waves_shift_vertical,
                    ],
                    outputs=[
                        image_effects_preview_spectrogram,
                    ],
                    scroll_to_output=True,
                )

            for slider_shuffle in [
                slider_shuffle_tile_size_x,
                slider_shuffle_tile_size_y,
                slider_shuffle_strength,
            ]:
                slider_shuffle.change(
                    fn=apply_shuffle,
                    inputs=[
                        state_effects_preview_spectrogram,
                        slider_shuffle_tile_size_x,
                        slider_shuffle_tile_size_y,
                        slider_shuffle_strength,
                    ],
                    outputs=[
                        image_effects_preview_spectrogram,
                    ],
                    scroll_to_output=True,
                )

            slider_threshold_value.change(
                fn=apply_threshold,
                inputs=[
                    state_effects_preview_spectrogram,
                    slider_threshold_value,
                ],
                outputs=[
                    image_effects_preview_spectrogram,
                ],
                scroll_to_output=True,
            )

            for slider_twirl in [
                slider_twirl_angle,
                slider_twirl_radius,
            ]:
                slider_twirl.change(
                    fn=apply_twirl,
                    inputs=[
                        state_effects_preview_spectrogram,
                        slider_twirl_angle,
                        slider_twirl_radius,
                    ],
                    outputs=[
                        image_effects_preview_spectrogram,
                    ],
                    scroll_to_output=True,
                )

            for slider_skew in [
                slider_skew_angle,
                slider_skew_factor,
            ]:
                slider_skew.change(
                    fn=apply_skew,
                    inputs=[
                        state_effects_preview_spectrogram,
                        slider_skew_angle,
                        slider_skew_factor,
                    ],
                    outputs=[
                        image_effects_preview_spectrogram,
                    ],
                    scroll_to_output=True,
                )

            slider_rotation_angle.change(
                fn=apply_rotation,
                inputs=[
                    state_effects_preview_spectrogram,
                    slider_rotation_angle,
                ],
                outputs=[
                    image_effects_preview_spectrogram,
                ],
                scroll_to_output=True,
            )

            for slider_scale in [
                slider_scale_factor_horizontal,
                slider_scale_factor_vertical,
            ]:
                slider_scale.change(
                    fn=apply_scale,
                    inputs=[
                        state_effects_preview_spectrogram,
                        slider_scale_factor_horizontal,
                        slider_scale_factor_vertical,
                    ],
                    outputs=[
                        image_effects_preview_spectrogram,
                    ],
                    scroll_to_output=True,
                )

            for slider_offset in [
                slider_offset_x,
                slider_offset_y,
            ]:
                slider_offset.change(
                    fn=apply_offset,
                    inputs=[
                        state_effects_preview_spectrogram,
                        slider_offset_x,
                        slider_offset_y,
                    ],
                    outputs=[
                        image_effects_preview_spectrogram,
                    ],
                    scroll_to_output=True,
                )

            slider_noise_value.change(
                fn=apply_noise,
                inputs=[
                    state_effects_preview_spectrogram,
                    slider_noise_value,
                ],
                outputs=[
                    image_effects_preview_spectrogram,
                ],
                scroll_to_output=True,
            )

            button_apply_image_effects_preview.click(
                fn=on_click_apply_image_effects_preview,
                inputs=[
                    image_effects_preview_spectrogram,
                ],
                outputs=[
                    image_editor_spectrogram,
                    state_effects_preview_spectrogram,
                ],
            )

            for slider_crop_offset in [
                slider_crop_offset_left,
                slider_crop_offset_upper,
                slider_crop_offset_right,
                slider_crop_offset_lower,
            ]:
                slider_crop_offset.change(
                    fn=apply_crop,
                    inputs=[
                        state_effects_preview_spectrogram,
                        slider_crop_offset_left,
                        slider_crop_offset_upper,
                        slider_crop_offset_right,
                        slider_crop_offset_lower,
                    ],
                    outputs=[
                        image_effects_preview_spectrogram,
                    ],
                    scroll_to_output=True,
                )

            state_tab_editor_id = gr.State(value=tab_editor.id)

            button_send_to_editor.click(
                fn=on_click_send_to_editor,
                inputs=[
                    image_output_spectrogram_preview,
                    state_tab_editor_id,
                ],
                outputs=[
                    image_editor_spectrogram,
                    image_effects_preview_spectrogram,
                    state_effects_preview_spectrogram,
                    tabs,
                ],
            )

            button_open_in_krita.click(
                fn=on_click_open_in_krita,
                inputs=[
                    textbox_output_spectrogram,
                    image_editor_spectrogram,
                ],
                outputs=[
                    image_editor_spectrogram,
                ],
            )

        with gr.Tab("From Spectrogram", id=3) as tab_to_audio:
            with gr.Row():
                with gr.Column():
                    file_input_spectrogram = gr.File(label="Input Spectrogram File")

                    slider_sample_rate_to_audio = gr.Slider(
                        label="Sample Rate",
                        minimum=0,
                        maximum=192000,
                        step=1,
                        value=SAMPLE_RATE,
                    )

                    slider_n_fft_to_audio = gr.Slider(
                        label="FFT Window Size",
                        minimum=0,
                        maximum=65536,
                        step=1,
                        value=N_FFT,
                    )

                    slider_hop_length_to_audio = gr.Slider(
                        label="FFT Window Hop Length",
                        minimum=0,
                        maximum=65536,
                        step=1,
                        value=HOP_LENGTH,
                    )

                    slider_pitch_shift_to_audio = gr.Slider(
                        label="Pitch Shift",
                        minimum=-48,
                        maximum=48,
                        step=0.1,
                        value=0,
                    )

                    slider_gain_to_audio = gr.Slider(
                        label="Gain",
                        minimum=0,
                        maximum=10.0,
                        step=0.01,
                        value=1.0,
                    )

                    slider_gain_multiplier_to_audio = gr.Radio(
                        label="Gain Multiplier",
                        choices=[1, 10, 100, 1000, 10000, 100000],
                        value=1,
                    )

                    slider_clip_to_audio = gr.Slider(
                        label="Clip",
                        minimum=0,
                        maximum=1.0,
                        step=0.01,
                        value=1.0,
                    )

                    button_convert_to_audio = gr.Button("Convert")

                with gr.Column():
                    textbox_output_audio = gr.Textbox(
                        label="Output Audio Path",
                        show_copy_button=True,
                        interactive=False,
                    )

                    audio_output_preview = gr.Audio(
                        visible=False,
                    )

                    code_output_audio_info = gr.Code(visible=False)

            state_tab_to_audio_id = gr.State(value=tab_to_audio.id)

            button_send_to_audio_from_editor.click(
                fn=on_click_send_to_audio_from_editor,
                inputs=[
                    textbox_output_spectrogram,
                    image_editor_spectrogram,
                    state_tab_to_audio_id,
                ],
                outputs=[
                    file_input_spectrogram,
                    tabs,
                ],
            )

            button_send_to_audio.click(
                fn=on_click_send_to_audio,
                inputs=[
                    textbox_output_spectrogram,
                    state_tab_to_audio_id,
                ],
                outputs=[
                    file_input_spectrogram,
                    tabs,
                ],
            )

            button_convert_to_audio.click(
                fn=on_click_to_audio,
                inputs=[
                    file_input_spectrogram,
                    slider_sample_rate_to_audio,
                    slider_n_fft_to_audio,
                    slider_hop_length_to_audio,
                    slider_pitch_shift_to_audio,
                    slider_gain_to_audio,
                    slider_gain_multiplier_to_audio,
                    slider_clip_to_audio,
                ],
                outputs=[
                    textbox_output_audio,
                    audio_output_preview,
                    code_output_audio_info,
                ],
            )

            file_input_audio.upload(
                fn=on_upload_input_audio,
                inputs=[
                    file_input_audio,
                    checkbox_trim_to_spectrogram,
                    slider_trim_threshold_to_spectrogram,
                ],
                outputs=[
                    audio_input_preview,
                    slider_sample_rate_to_spectrogram,
                    slider_sample_rate_to_audio,
                    code_input_audio_info,
                    button_convert_to_spectrogram,
                ],
            )

            slider_trim_threshold_to_spectrogram.change(
                fn=on_change_trim_threshold_to_spectrogram,
                inputs=[
                    file_input_audio,
                    slider_trim_threshold_to_spectrogram,
                ],
                outputs=[
                    audio_input_preview,
                    code_input_audio_info,
                ],
            )

blocks.launch()

