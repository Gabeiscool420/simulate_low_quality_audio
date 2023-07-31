import os
import random
import librosa
import numpy as np
from scipy.signal import butter, iirfilter, lfilter
import soundfile as sf
from pydub import AudioSegment
import pyroomacoustics as pra

# Function to apply a Butterworth filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to apply a peaking equalizer filter
def apply_eq(data, center_freq, Q, gain, fs):
    freqs = sorted([center_freq * Q, center_freq / Q])  # Sort the frequencies
    b, a = iirfilter(2, freqs, rs=1, btype='band', analog=False, ftype='butter', fs=fs)
    y = lfilter(b, a, data)
    y = y * gain
    return y

# Function to get a random audio file from a directory
def get_random_audio_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    return random.choice(files)

# Function to get a random segment from an audio file
def get_random_segment(filename, segment_duration):
    full_audio, sr = librosa.load(filename, sr=None)
    full_duration = librosa.get_duration(y=full_audio, sr=sr)
    random_offset = random.uniform(0, full_duration - segment_duration)
    segment, _ = librosa.load(filename, sr=None, offset=random_offset, duration=segment_duration)
    return segment, sr

# Function to loop and crossfade an audio signal to match a target length
def loop_and_crossfade(signal, target_length, crossfade_duration):
    crossfade_samples = int(crossfade_duration * sr)
    looped_signal = np.copy(signal)
    while len(looped_signal) < target_length:
        looped_signal = np.concatenate((looped_signal[:-crossfade_samples], looped_signal, signal[crossfade_samples:]))
    return looped_signal[:target_length]

# Function to apply a short random reverb
def apply_reverb(signal, sr, rt60):
    # Create a shoebox room
    room_dim = np.random.uniform(5, 10, size=(3,))  # Random room dimensions in meters
    V = np.prod(room_dim)  # volume of the room
    S = 2 * (room_dim[0]*room_dim[1] + room_dim[1]*room_dim[2] + room_dim[0]*room_dim[2])  # surface area
    alpha = 0.1611 * V / (S * rt60)  # absorption coefficient
    room = pra.ShoeBox(room_dim, fs=sr, max_order=15, absorption={'east': alpha, 'west': alpha, 'north': alpha, 'south': alpha, 'floor': alpha, 'ceiling': alpha})

    # Add the source and the microphone to the room
    room.add_source([2, 2, 2], signal=signal)
    room.add_microphone_array(pra.MicrophoneArray(np.array([[4, 4, 2]]).T, fs=room.fs))

    # Compute the room impulse response
    room.compute_rir()

    # Convolve the source signal with the room impulse response
    room.simulate()
    return room.mic_array.signals[0, :]

# Function to calculate the RMS level of a signal
def rms_level(signal, window_size):
    # Calculate the square of the signal
    signal_squared = np.power(signal, 2)
    # Apply the moving average filter
    rms_signal = np.sqrt(np.convolve(signal_squared, np.ones(window_size) / window_size, 'same'))
    return rms_signal

# Function to apply dynamic compression
def apply_dynamic_compression(signal, side_chain, window_size, threshold, ratio, attack, release, sr):
    # Calculate the RMS level of the side-chain signal
    rms_side_chain = rms_level(side_chain, window_size)
    # Convert the threshold and the RMS level of the side-chain signal to dB
    threshold_db = 20 * np.log10(threshold)
    rms_side_chain_db = 20 * np.log10(rms_side_chain + np.finfo(float).eps)  # add eps to avoid log(0)
    # Calculate the gain reduction in dB
    gain_reduction_db = np.maximum(rms_side_chain_db - threshold_db, 0) * (ratio - 1)
    # Apply the attack and release times
    for i in range(1, len(gain_reduction_db)):
        if gain_reduction_db[i] > gain_reduction_db[i - 1]:
            gain_reduction_db[i] = gain_reduction_db[i - 1] + 1 / (attack * sr)
        else:
            gain_reduction_db[i] = gain_reduction_db[i - 1] - 1 / (release * sr)
    # Convert the gain reduction from dB to linear scale
    gain_reduction = np.power(10, -gain_reduction_db / 20)
    # Apply the gain reduction to the signal
    compressed_signal = signal * gain_reduction
    return compressed_signal

# Load the high-quality audio
filename = 'guit.wav'
y, sr = librosa.load(filename, sr=None)

# Apply a random EQ boost
center_freq = random.uniform(640, 1280)  # Center frequency for the EQ boost
Q = random.uniform(0.3, 0.48)  # Q factor for the EQ boost
gain = random.uniform(0.55, 0.85)  # Gain for the EQ boost
y_noisy = apply_eq(y, center_freq, Q, gain, sr)

# Apply a second EQ boost to simulate the proximity effect
proximity_center_freq = random.uniform(350, 400)  # Center frequency for the proximity effect EQ boost
proximity_Q = 1 / gain  # Q factor for the proximity effect EQ boost, inversely proportional to the first gain
proximity_gain = gain + 0.5  # Gain for the proximity effect EQ boost, equal to the first gain
y_noisy = apply_eq(y_noisy, proximity_center_freq, proximity_Q, proximity_gain, sr)

# Apply a short random reverb
rt60 = random.uniform(0.17, 0.23)  # Random reverberation time between 0.1 and 0.3 seconds
y_noisy = apply_reverb(y_noisy, sr, rt60)

# Add noise to simulate the noise typically introduced by phone microphones
noise_amp = 0.00075  # Reduced noise level
y_noisy = y_noisy + noise_amp * np.random.normal(size=len(y_noisy))

# Load a random room sound
room_directory = 'rooms'
room_filename = get_random_audio_file(room_directory)
segment_duration = librosa.get_duration(y=y, sr=sr)  # Set segment duration to the duration of the high-quality track
room_sound, _ = get_random_segment(os.path.join(room_directory, room_filename), segment_duration)

# Adjust the volume of the room sound
room_sound = room_sound * 0.5  # Adjust volume here, 0.5 for example

# Loop and crossfade the room sound to match the length of the high-quality track
room_sound = loop_and_crossfade(room_sound, len(y_noisy), 0.5)  # Crossfade duration of 0.5 seconds

# Add the room sound to the simulated low-quality audio
y_room = y_noisy + room_sound

# Apply dynamic compression
window_size = int(0.1 * sr)  # Window size for the RMS level calculation (0.1 seconds)
threshold = 0.05  # Threshold for the compression (in linear scale)
ratio = 4  # Compression ratio
attack = 0.01  # Attack time (in seconds)
release = 0.1  # Release time (in seconds)
room_sound_compressed = apply_dynamic_compression(room_sound, y_noisy, window_size, threshold, ratio, attack, release, sr)

# Add the compressed room sound to the simulated low-quality audio
y_room_compressed = y_noisy + room_sound_compressed

# Apply the bandpass filter to simulate the limited frequency response of a phone microphone
random_lowcut = random.uniform(315, 335)  # Low frequency limit (in Hz)
random_highcut = random.uniform(3325, 3375)  # High frequency limit (in Hz)
y_filtered_compressed = butter_bandpass_filter(y_room_compressed, random_lowcut, random_highcut, sr, order=6)

# Save the noisy audio to a temporary file
temp_filename_compressed = 'temp_compressed.wav'
sf.write(temp_filename_compressed, y_filtered_compressed, sr)

# Load the audio with pydub
audio_compressed = AudioSegment.from_wav(temp_filename_compressed)

# Apply a gain of 6dB
audio_compressed = audio_compressed.apply_gain(18)

# Convert the audio to MP3 and back to WAV to simulate compression artifacts
compressed_filename = "temp_compressed.mp3"
audio_compressed.export(compressed_filename, format="mp3")
compressed_audio_compressed = AudioSegment.from_mp3(compressed_filename)
final_filename = 'simulated_low_quality_audio_compressed.wav'
compressed_audio_compressed.export(final_filename, format="wav")
