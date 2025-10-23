"""
Audio Engine for BAPS-1 Sampler
Handles audio loading, playback, mixing, and pitch shifting
"""

import math
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Dict, Optional, List, Set
import threading
from dataclasses import dataclass


@dataclass
class Voice:
    """Represents an active playing voice"""
    audio_data: np.ndarray  # The audio data to play
    slot_id: Optional[int] = None  # Which slot/channel this voice belongs to (1-16)
    choke_group: int = 0  # Choke group (0=polyphonic, 1-15=monophonic group)
    position: float = 0.0  # Current playback position in samples
    volume: float = 1.0  # Volume multiplier
    speed: float = 1.0  # Playback speed (for pitch shifting)
    is_active: bool = True
    is_fading_out: bool = False  # Triggered when cut by another note
    fade_out_position: int = 0  # Position in fade-out envelope
    fade_samples: int = 2205  # 100ms fade at 22050Hz

    def get_next_samples(self, num_samples: int) -> np.ndarray:
        """Get the next chunk of samples with fade-in/fade-out"""
        if not self.is_active:
            return np.zeros(num_samples)

        # Calculate how many samples we can get
        samples_remaining = len(self.audio_data) - int(self.position)
        if samples_remaining <= 0:
            self.is_active = False
            return np.zeros(num_samples)

        # Get samples with speed adjustment (simple resampling)
        output = np.zeros(num_samples)
        for i in range(num_samples):
            if self.position >= len(self.audio_data):
                self.is_active = False
                break

            # Linear interpolation for smoother resampling
            idx = int(self.position)
            frac = self.position - idx

            if idx + 1 < len(self.audio_data):
                sample = self.audio_data[idx] * (1 - frac) + self.audio_data[idx + 1] * frac
            else:
                sample = self.audio_data[idx]

            # Apply fade-in/fade-out envelope
            envelope = 1.0

            # If voice is fading out (cut by another note)
            if self.is_fading_out:
                # Fade out from current position with cosine curve (smoother)
                if self.fade_out_position < self.fade_samples:
                    # Cosine fade: starts at 1.0, smoothly goes to 0.0
                    fade_progress = self.fade_out_position / self.fade_samples
                    envelope = 0.5 * (1.0 + np.cos(fade_progress * np.pi))
                    self.fade_out_position += 1
                else:
                    # Fade complete
                    self.is_active = False
                    break
            else:
                # No fade-in at start - sound starts immediately

                # Fade-out at natural end with cosine curve
                samples_from_end = len(self.audio_data) - idx
                if samples_from_end < self.fade_samples:
                    fade_progress = samples_from_end / self.fade_samples
                    # Cosine fade-out: starts at 1.0, smoothly goes to 0.0
                    envelope = 0.5 * (1.0 + np.cos((1.0 - fade_progress) * np.pi))

            output[i] = sample * self.volume * envelope
            self.position += self.speed

        return output


class AudioEngine:
    """Main audio engine for sample playback and mixing"""

    def __init__(self, sample_rate: int = 22050, blocksize: int = 512):
        self.sample_rate = sample_rate
        self.blocksize = blocksize

        # Cache for loaded audio files
        self.audio_cache: Dict[str, np.ndarray] = {}

        # Active voices
        self.voices: List[Voice] = []
        self.voice_lock = threading.Lock()

        # Output stream
        self.stream: Optional[sd.OutputStream] = None
        self.is_running = False

        # Master volume
        self.master_volume = 1.0

        # FX coordination
        self.fx_lock = threading.RLock()
        self._active_effect = "none"

        # Global FX filter state
        self._fx_filter_high_pass = 0.0
        self._fx_filter_low_pass = 0.0
        self._fx_filter_hp_prev_x = 0.0
        self._fx_filter_hp_prev_y = 0.0
        self._fx_filter_lp_prev_y = 0.0
        self._fx_filter_enabled = False

        # Reverb state
        reverb_times = (0.0297, 0.0371, 0.0411, 0.0437)
        self._reverb_delay_lengths = [
            max(1, int(self.sample_rate * t)) for t in reverb_times
        ]
        self._reverb_buffers = [
            np.zeros(length, dtype=np.float32) for length in self._reverb_delay_lengths
        ]
        self._reverb_indices = [0] * len(self._reverb_delay_lengths)
        self._reverb_mix = 0.0
        self._reverb_decay = 0.0
        self._reverb_feedback = 0.5

        # Chorus state
        self._chorus_depth = 0.0
        self._chorus_rate = 0.0
        self._chorus_phase = 0.0
        self._chorus_buffer_length = max(1, int(self.sample_rate * 0.05))
        self._chorus_buffer = np.zeros(self._chorus_buffer_length, dtype=np.float32)
        self._chorus_write_index = 0

        # Pattern timing & one-shot FX
        self._pattern_samples = 0
        self._pattern_position = 0
        self._beat_samples = 0
        self._repeat_enabled_for_pattern = False
        self._oneshot_active = set()
        self._repeat_buffer = np.zeros(1, dtype=np.float32)

        # Vinyl stop buffers
        self._vinyl_buffer = np.zeros(max(1, int(self.sample_rate * 6)), dtype=np.float32)
        self._vinyl_write_pos = 0
        self._vinyl_read_pos = 0.0
        self._vinyl_active = False
        self._vinyl_elapsed = 0

    def start(self):
        """Start the audio engine"""
        if self.is_running:
            return

        self.is_running = True
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()
        print(f"Audio engine started (sample rate: {self.sample_rate}Hz)")

    def stop(self):
        """Stop the audio engine"""
        if not self.is_running:
            return

        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Audio engine stopped")

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback for sounddevice"""
        if status:
            print(f"Audio status: {status}")

        # Mix all active voices
        mixed = np.zeros(frames)

        with self.voice_lock:
            # Remove inactive voices
            self.voices = [v for v in self.voices if v.is_active]

            # Mix active voices
            for voice in self.voices:
                mixed += voice.get_next_samples(frames)

        with self.fx_lock:
            processed = self._apply_permanent_effect(mixed)
            processed = self._apply_one_shot_effects(processed)

        # Apply master volume and clip
        processed *= self.master_volume
        processed = np.clip(processed, -1.0, 1.0)

        # Write to output
        outdata[:, 0] = processed

    def set_filter_params(self, high_pass_amount: float, low_pass_amount: float):
        """
        Update the global filter effect parameters.

        Args:
            high_pass_amount: High-pass amount (0.0-1.0)
            low_pass_amount: Low-pass amount (0.0-1.0)
        """
        hp = max(0.0, min(1.0, float(high_pass_amount)))
        lp = max(0.0, min(1.0, float(low_pass_amount)))

        with self.fx_lock:
            self._fx_filter_high_pass = hp
            self._fx_filter_low_pass = lp
            self._fx_filter_enabled = (hp > 0.001) or (lp > 0.001)
            if self._fx_filter_enabled:
                self._active_effect = "filter"
            elif self._active_effect == "filter":
                self._active_effect = "none"

            if not self._fx_filter_enabled:
                # Reset filter state when disabling to avoid stale responses
                self._fx_filter_hp_prev_x = 0.0
                self._fx_filter_hp_prev_y = 0.0
                self._fx_filter_lp_prev_y = 0.0

    def _apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Apply active filters to the audio buffer."""
        processed = audio

        if self._fx_filter_high_pass > 0.0:
            processed = self._apply_high_pass(processed, self._fx_filter_high_pass)

        if self._fx_filter_low_pass > 0.0:
            processed = self._apply_low_pass(processed, self._fx_filter_low_pass)

        return processed

    def _apply_high_pass(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Simple first-order high-pass filter with eased control response."""
        cutoff = self._map_cutoff(amount, 40.0, 5000.0)
        cutoff = max(10.0, min(cutoff, self.sample_rate / 2 - 100.0))

        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / self.sample_rate
        alpha = rc / (rc + dt)

        prev_y = self._fx_filter_hp_prev_y
        prev_x = self._fx_filter_hp_prev_x

        out = np.empty_like(audio)
        for i, x in enumerate(audio):
            y = alpha * (prev_y + x - prev_x)
            out[i] = y
            prev_y = y
            prev_x = x

        self._fx_filter_hp_prev_y = prev_y
        self._fx_filter_hp_prev_x = prev_x
        return out

    def _apply_low_pass(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Simple first-order low-pass filter with eased control response."""
        # Reverse mapping so higher amount lowers cutoff
        cutoff = self._map_cutoff(1.0 - amount, 200.0, 12000.0)
        cutoff = max(50.0, min(cutoff, self.sample_rate / 2 - 100.0))

        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / self.sample_rate
        alpha = dt / (rc + dt)

        prev_y = self._fx_filter_lp_prev_y
        out = np.empty_like(audio)
        for i, x in enumerate(audio):
            prev_y = prev_y + alpha * (x - prev_y)
            out[i] = prev_y

        self._fx_filter_lp_prev_y = prev_y
        return out

    @staticmethod
    def _map_cutoff(amount: float, min_freq: float, max_freq: float) -> float:
        """Map normalized control value to frequency using quadratic response."""
        curved = amount * amount
        return min_freq + (max_freq - min_freq) * curved

    def set_effect_none(self):
        """Disable any active global effect."""
        with self.fx_lock:
            self._active_effect = "none"
            self._fx_filter_enabled = False
            self._fx_filter_hp_prev_x = 0.0
            self._fx_filter_hp_prev_y = 0.0
            self._fx_filter_lp_prev_y = 0.0
            self._reverb_mix = 0.0
            self._reverb_decay = 0.0
            self._reset_reverb_state()
            self._chorus_depth = 0.0
            self._chorus_rate = 0.0
            self._reset_chorus_state()

    def start_pattern_cycle(self, pattern_samples: int, beat_samples: int, active_pads: Set[int]):
        """Notify the engine that a new pattern cycle has started."""
        beat_samples = max(1, beat_samples)
        pattern_samples = max(beat_samples * 4, pattern_samples)

        with self.fx_lock:
            self._oneshot_active = set(active_pads)
            self._pattern_samples = pattern_samples
            self._beat_samples = beat_samples
            self._pattern_position = 0
            self._repeat_enabled_for_pattern = 1 in self._oneshot_active
            self._ensure_repeat_buffer_capacity(beat_samples)
            if self._repeat_enabled_for_pattern:
                self._repeat_buffer.fill(0.0)

            self._ensure_vinyl_buffer_capacity(pattern_samples + beat_samples)
            if 2 in self._oneshot_active:
                self._vinyl_active = True
                self._vinyl_elapsed = 0
                self._vinyl_read_pos = float(self._vinyl_write_pos)
            else:
                self._vinyl_active = False
                self._vinyl_elapsed = 0
                self._vinyl_read_pos = float(self._vinyl_write_pos)
    def set_reverb_params(self, mix_amount: float, decay_amount: float):
        """
        Configure the global reverb effect.

        Args:
            mix_amount: Wet/dry mix (0.0-1.0)
            decay_amount: Decay length (0.0-1.0)
        """
        mix = max(0.0, min(1.0, float(mix_amount)))
        decay = max(0.0, min(1.0, float(decay_amount)))

        with self.fx_lock:
            was_active = (self._active_effect == "reverb" and
                          self._reverb_mix > 0.001 and
                          self._reverb_decay > 0.001)

            self._reverb_mix = mix
            self._reverb_decay = decay

            enabled = mix > 0.001 and decay > 0.001

            if enabled:
                self._reverb_feedback = 0.3 + decay * 0.6  # Stable feedback (<1.0)
                if self._active_effect != "reverb" or not was_active:
                    self._reset_reverb_state()
                self._active_effect = "reverb"
            else:
                if self._active_effect == "reverb":
                    self._active_effect = "none"
                self._reset_reverb_state()

    def set_chorus_params(self, depth_amount: float, rate_amount: float):
        """
        Configure the global chorus effect.

        Args:
            depth_amount: Modulation depth / mix (0.0-1.0)
            rate_amount: LFO speed (0.0-1.0)
        """
        depth = max(0.0, min(1.0, float(depth_amount)))
        rate = max(0.0, min(1.0, float(rate_amount)))

        with self.fx_lock:
            self._chorus_depth = depth
            self._chorus_rate = rate
            enabled = depth > 0.001

            if enabled:
                if self._active_effect != "chorus":
                    self._reset_chorus_state()
                self._active_effect = "chorus"
            else:
                if self._active_effect == "chorus":
                    self._active_effect = "none"
                self._reset_chorus_state()

    def _apply_permanent_effect(self, audio: np.ndarray) -> np.ndarray:
        """Dispatch audio through the currently selected permanent effect."""
        effect = self._active_effect

        if effect == "filter" and self._fx_filter_enabled:
            return self._apply_filters(audio)
        if effect == "reverb" and self._reverb_mix > 0.001 and self._reverb_decay > 0.001:
            return self._apply_reverb(audio)
        if effect == "chorus" and self._chorus_depth > 0.001:
            return self._apply_chorus(audio)
        return audio

    def _apply_one_shot_effects(self, audio: np.ndarray) -> np.ndarray:
        """Apply one-shot FX (beat repeat, vinyl stop) on top of the mixed signal."""
        if self._pattern_samples <= 0 or self._beat_samples <= 0:
            return audio

        length = len(audio)
        pattern_len = self._pattern_samples
        beat_len = self._beat_samples
        buffer_len = self._vinyl_buffer.shape[0]

        output = np.empty_like(audio)
        write_pos = self._vinyl_write_pos
        read_pos = self._vinyl_read_pos
        base_elapsed = self._vinyl_elapsed

        repeat_enabled = self._repeat_enabled_for_pattern
        vinyl_enabled = self._vinyl_active

        for i in range(length):
            pattern_pos = (self._pattern_position + i) % pattern_len
            source_sample = audio[i]
            sample = source_sample

            if repeat_enabled:
                if pattern_pos < beat_len:
                    self._repeat_buffer[pattern_pos] = source_sample
                else:
                    sample = self._repeat_buffer[pattern_pos % beat_len]

            # write post-repeat signal into vinyl buffer
            self._vinyl_buffer[write_pos] = sample
            write_pos = (write_pos + 1) % buffer_len

            if vinyl_enabled:
                progress = min(1.0, (base_elapsed + i) / max(1, pattern_len))
                speed = max(0.0, (1.0 - progress) ** 2)
                read_pos += speed
                while read_pos >= buffer_len:
                    read_pos -= buffer_len
                idx0 = int(read_pos)
                idx1 = (idx0 + 1) % buffer_len
                frac = read_pos - idx0
                vinyl_sample = (1.0 - frac) * self._vinyl_buffer[idx0] + frac * self._vinyl_buffer[idx1]
                envelope = max(0.0, (1.0 - progress) ** 1.5)
                sample = vinyl_sample * envelope

            output[i] = sample

        self._vinyl_write_pos = write_pos
        if vinyl_enabled:
            self._vinyl_read_pos = read_pos
            self._vinyl_elapsed = base_elapsed + length
            if self._vinyl_elapsed >= pattern_len:
                self._vinyl_active = False
        else:
            self._vinyl_read_pos = float(self._vinyl_write_pos)
            self._vinyl_elapsed = 0

        self._pattern_position = (self._pattern_position + length) % pattern_len
        return output

    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply a lightweight Schroeder-style reverb."""
        if self._reverb_mix <= 0.001 or self._reverb_decay <= 0.001:
            return audio

        buffers = self._reverb_buffers
        indices = self._reverb_indices
        num_lines = len(buffers)
        feedback = min(0.95, self._reverb_feedback)
        mix = min(0.9, self._reverb_mix)

        output = np.empty_like(audio)

        for i, sample in enumerate(audio):
            accum = 0.0
            for line_idx in range(num_lines):
                buf = buffers[line_idx]
                idx = indices[line_idx]
                delayed = float(buf[idx])
                buf[idx] = sample + delayed * feedback
                indices[line_idx] = (idx + 1) % buf.size
                accum += delayed

            wet = accum / num_lines
            dry = 1.0 - mix
            output[i] = dry * sample + mix * wet

        return output

    def _apply_chorus(self, audio: np.ndarray) -> np.ndarray:
        """Apply a dual-tap chorus with sine LFO."""
        depth = self._chorus_depth
        if depth <= 0.001:
            return audio

        rate_hz = self._map_chorus_rate(self._chorus_rate)
        buffer = self._chorus_buffer
        buffer_len = self._chorus_buffer_length
        write_idx = self._chorus_write_index
        phase = self._chorus_phase

        base_delay = 0.015 * self.sample_rate  # 15 ms base
        mod_depth = depth * (0.010 * self.sample_rate)  # up to +10 ms
        mix = min(0.8, 0.3 + depth * 0.5)

        output = np.empty_like(audio)

        for i, sample in enumerate(audio):
            buffer[write_idx] = sample

            lfo = 0.5 * (1.0 + math.sin(phase))
            delay = base_delay + lfo * mod_depth
            read_pos = write_idx - delay
            while read_pos < 0:
                read_pos += buffer_len

            idx0 = int(read_pos)
            idx1 = (idx0 + 1) % buffer_len
            frac = read_pos - idx0
            delayed = (1.0 - frac) * buffer[idx0] + frac * buffer[idx1]

            dry = 1.0 - mix
            output[i] = dry * sample + mix * delayed

            write_idx = (write_idx + 1) % buffer_len
            phase += (2.0 * math.pi * rate_hz) / self.sample_rate
            if phase >= 2.0 * math.pi:
                phase -= 2.0 * math.pi

        self._chorus_write_index = write_idx
        self._chorus_phase = phase
        return output

    def _reset_reverb_state(self):
        """Clear reverb buffers and indices."""
        for buf in self._reverb_buffers:
            buf.fill(0.0)
        self._reverb_indices = [0] * len(self._reverb_buffers)

    def _reset_chorus_state(self):
        """Clear chorus modulation history."""
        self._chorus_buffer.fill(0.0)
        self._chorus_write_index = 0
        self._chorus_phase = 0.0

    @staticmethod
    def _map_chorus_rate(rate_amount: float) -> float:
        """Map normalized knob value to an LFO rate in Hz."""
        return 0.1 + rate_amount * 4.9

    def _ensure_repeat_buffer_capacity(self, length: int):
        length = max(1, int(length))
        if self._repeat_buffer.shape[0] != length:
            self._repeat_buffer = np.zeros(length, dtype=np.float32)

    def _ensure_vinyl_buffer_capacity(self, required_samples: int):
        required_samples = max(required_samples, self._vinyl_buffer.shape[0])
        if required_samples > self._vinyl_buffer.shape[0]:
            new_size = max(required_samples, self._vinyl_buffer.shape[0] * 2)
            self._vinyl_buffer = np.zeros(new_size, dtype=np.float32)
            self._vinyl_write_pos = 0
            self._vinyl_read_pos = 0.0

    def load_audio(self, filepath: str) -> Optional[np.ndarray]:
        """Load an audio file and cache it"""
        if filepath in self.audio_cache:
            return self.audio_cache[filepath]

        try:
            # Load audio file
            audio_data, file_samplerate = sf.read(filepath)

            # Convert stereo to mono if needed
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if needed (simple decimation/interpolation)
            if file_samplerate != self.sample_rate:
                audio_data = self._resample(audio_data, file_samplerate, self.sample_rate)

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.9  # Leave some headroom

            # Cache it
            self.audio_cache[filepath] = audio_data
            print(f"Loaded audio: {filepath} ({len(audio_data)} samples)")
            return audio_data

        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}")
            return None

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple audio resampling using linear interpolation"""
        if from_rate == to_rate:
            return audio

        # Calculate new length
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)

        # Create new sample positions
        old_positions = np.arange(len(audio))
        new_positions = np.linspace(0, len(audio) - 1, new_length)

        # Interpolate
        resampled = np.interp(new_positions, old_positions, audio)
        return resampled

    def play_sound(self, filepath: str, volume: float = 1.0, pitch_shift: int = 0, slot_id: Optional[int] = None, choke_group: int = 0):
        """
        Play a sound with optional pitch shifting

        Args:
            filepath: Path to the audio file
            volume: Volume multiplier (0.0 to 1.0)
            pitch_shift: Semitones to shift (-12 to +12)
            slot_id: Slot/channel number (1-16) for monophonic playback
            choke_group: Choke group (0=polyphonic, 1-15=monophonic group)
        """
        audio_data = self.load_audio(filepath)
        if audio_data is None:
            return

        # Calculate speed from pitch shift (12 semitones = 2x speed)
        speed = 2 ** (pitch_shift / 12.0)

        with self.voice_lock:
            # Handle choke groups
            if choke_group > 0:
                # Fade out all voices in the same choke group
                for v in self.voices:
                    if v.choke_group == choke_group:
                        v.is_fading_out = True
            elif slot_id is not None:
                # Legacy behavior: fade out voices from same slot
                for v in self.voices:
                    if v.slot_id == slot_id:
                        v.is_fading_out = True

            # Create and add voice
            # No need to copy - Voice only reads from audio_data
            voice = Voice(
                audio_data=audio_data,
                slot_id=slot_id,
                choke_group=choke_group,
                volume=volume,
                speed=speed
            )
            self.voices.append(voice)

    def play_slice(self, filepath: str, start_time: float, duration: float, volume: float = 1.0, slot_id: Optional[int] = None, pitch_shift: float = 0, choke_group: int = 0):
        """
        Play a slice of an audio file

        Args:
            filepath: Path to the audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            volume: Volume multiplier (0.0 to 1.0)
            slot_id: Slot/channel number (1-16) for monophonic playback
            pitch_shift: Semitones to shift (-12.0 to +12.0)
            choke_group: Choke group (0=polyphonic, 1-15=monophonic group)
        """
        audio_data = self.load_audio(filepath)
        if audio_data is None or duration <= 0:
            return

        # Convert time to samples
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)

        # Bounds check
        start_sample = max(0, min(start_sample, len(audio_data) - 1))
        end_sample = max(start_sample + 1, min(end_sample, len(audio_data)))

        # Extract slice (use view, not copy for better performance)
        sliced_audio = audio_data[start_sample:end_sample]

        # Calculate speed from pitch shift (12 semitones = 2x speed)
        speed = 2 ** (pitch_shift / 12.0)

        with self.voice_lock:
            # Handle choke groups
            if choke_group > 0:
                # Fade out all voices in the same choke group
                for v in self.voices:
                    if v.choke_group == choke_group:
                        v.is_fading_out = True
            elif slot_id is not None:
                # Legacy behavior: fade out voices from same slot
                for v in self.voices:
                    if v.slot_id == slot_id:
                        v.is_fading_out = True

            # Create and add voice
            voice = Voice(
                audio_data=sliced_audio,
                slot_id=slot_id,
                choke_group=choke_group,
                volume=volume,
                speed=speed
            )
            self.voices.append(voice)

    def stop_all(self):
        """Stop all currently playing sounds"""
        with self.voice_lock:
            self.voices.clear()

    def get_active_voice_count(self) -> int:
        """Get the number of currently active voices"""
        with self.voice_lock:
            return len(self.voices)

    def get_current_level(self) -> float:
        """Get the current audio output level (RMS)"""
        with self.voice_lock:
            if not self.voices:
                return 0.0

            # Get a small sample from each voice and calculate RMS
            sample_size = 512
            output = np.zeros(sample_size)

            for voice in self.voices:
                if voice.is_active:
                    # Get samples without advancing position
                    saved_pos = voice.position
                    samples = voice.get_next_samples(sample_size)
                    voice.position = saved_pos  # Restore position
                    output += samples

            # Calculate RMS
            rms = np.sqrt(np.mean(output ** 2))
            return min(1.0, float(rms) * 3.0)  # Amplify a bit for visibility

    def clear_cache(self):
        """Clear the audio cache"""
        self.audio_cache.clear()
        print("Audio cache cleared")

    def preload_sounds(self, filepaths: List[str]):
        """Preload multiple sound files into cache"""
        for filepath in filepaths:
            self.load_audio(filepath)

    def get_audio_duration(self, filepath: str) -> float:
        """Get the duration of an audio file in seconds"""
        audio_data = self.load_audio(filepath)
        if audio_data is None:
            return 0.0
        return len(audio_data) / self.sample_rate

    def get_audio_level(self, filepath: str, start_time: float = 0, duration: float = None) -> float:
        """
        Get the RMS level of an audio file or slice

        Args:
            filepath: Path to the audio file
            start_time: Start time in seconds (default: 0)
            duration: Duration in seconds (default: entire file)

        Returns:
            RMS level (0.0 to 1.0)
        """
        audio_data = self.load_audio(filepath)
        if audio_data is None:
            return 0.0

        # Get slice
        start_sample = int(start_time * self.sample_rate)
        if duration is None:
            end_sample = len(audio_data)
        else:
            end_sample = int((start_time + duration) * self.sample_rate)

        start_sample = max(0, min(start_sample, len(audio_data) - 1))
        end_sample = max(start_sample + 1, min(end_sample, len(audio_data)))

        audio_slice = audio_data[start_sample:end_sample]

        # Calculate RMS
        rms = np.sqrt(np.mean(audio_slice ** 2))
        return float(rms)
