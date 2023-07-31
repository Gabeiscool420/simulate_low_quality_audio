# simulate_low_quality_audio
---

# Low Quality Audio Simulation Module

This Python module provides functionality to simulate low-quality audio data. It is particularly useful for data augmentation tasks in audio processing. The module includes a main function, `simulate_low_quality_audio()`, which applies various transformations to high-quality audio to produce simulated low-quality audio. These transformations include:

- Applying a random EQ boost
- Simulating the proximity effect with a second EQ boost
- Adding a short random reverb
- Simulating noise typically introduced by phone microphones
- Adding a random room sound
- Applying dynamic compression
- Simulating the limited frequency response of a phone microphone
- Simulating compression artifacts by converting the audio to MP3 and back to WAV

## Dependencies

The module requires the following Python libraries:

- `os`
- `random`
- `librosa`
- `numpy`
- `scipy`
- `soundfile`
- `pydub`
- `pyroomacoustics`

## Usage

First, import the module using the following line of code:

```bash
from audio_simulation import simulate_low_quality_audio
```

Then, call the `simulate_low_quality_audio()` function with the required parameters:

```bash
simulate_low_quality_audio('input.wav', 'room_sounds_directory', 'output.wav')
```

The function takes three parameters:

- `filename`: The path to the high-quality input audio file (in WAV format)
- `room_directory`: The path to the directory containing room sound files (also in WAV format)
- `output_filename`: The path where the output file will be saved (in WAV format)

The function will apply the aforementioned transformations to the input audio file and save the resulting low-quality audio to the output file.

## Contributing

Contributions to this project are welcome! Please submit a pull request or open an issue to propose changes or additions.
