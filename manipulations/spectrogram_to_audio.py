#!/usr/bin/env python3

# TODO: Fix module input.

from .constants import SAMPLE_RATE, \
                       N_FFT, \
                       HOP_LENGTH, \
                       IMAGE_FORMAT, \
                       SOUNDFILE_SUBTYPE

import numpy as np

import click
import librosa
import soundfile

from PIL import Image

@click.command()
@click.option("--input", help="Input file path.")
@click.option("--output", help="Output file path.")
@click.option("--sampling-rate", help="Sampling rate of output file.", default=SAMPLE_RATE)
@click.option("--n-fft", help="FFT window size.", default=N_FFT)
@click.option("--hop-length", help="FFT window hop length.", default=HOP_LENGTH)
def spectrogram_to_audio_command(
    input_path: str,
    output_path: str,
    sample_rate: float,
    n_fft: int,
    hop_length: int,
) -> None:
    spectrogram_to_audio(
        input_path,
        output_path,
        sample_rate,
        n_fft,
        hop_length,
    )

def spectrogram_to_audio(
    input_path: str,
    output_path: str,
    sample_rate: float,
    n_fft: int,
    hop_length: int,
) -> None:
    image = Image.open(fp=input_path).convert(IMAGE_FORMAT)
    image_data = np.array(image)

    audio_data = librosa.feature.inverse.mel_to_audio(
        image_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    soundfile.write(
        file=output_path,
        data=audio_data,
        samplerate=sample_rate,
        subtype=SOUNDFILE_SUBTYPE,
    )

if __name__ == "__main__":
    spectrogram_to_audio_command()
