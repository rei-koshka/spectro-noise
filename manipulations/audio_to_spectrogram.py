#!/usr/bin/env python3

# TODO: Fix module input.

from .constants import SAMPLE_RATE, \
                       N_FFT, \
                       N_MELS, \
                       HOP_LENGTH, \
                       IMAGE_FORMAT

import click
import librosa

from PIL import Image

@click.command()
@click.option("--input", help="Input file path.")
@click.option("--output", help="Output file path.")
@click.option("--sampling-rate", help="Sampling rate of input file.", default=SAMPLE_RATE)
@click.option("--n-fft", help="FFT window size.", default=N_FFT)
@click.option("--hop-length", help="FFT window hop length.", default=HOP_LENGTH)
@click.option("--n-mels", help="Mel scale size.", default=N_MELS)
def audio_to_spectrogram_command(
    input_path: str,
    output_path: str,
    sample_rate: float,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> None:
    audio_to_spectrogram(
        input_path,
        output_path,
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
    )

def audio_to_spectrogram(
    input_path: str,
    output_path: str,
    sample_rate: float,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> None:
    audio_data, sample_rate = librosa.load(input_path, sr=sample_rate)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
    )

    image = Image.fromarray(mel_spectrogram).convert(IMAGE_FORMAT)

    image.save(output_path)

if __name__ == "__main__":
    audio_to_spectrogram_command()
