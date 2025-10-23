"""
State Manager for BAPS-1 Sampler
Handles all application state, settings, and data structures
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import os
from typing import Tuple

import numpy as np
import soundfile as sf


class Mode(Enum):
    """Operating modes for the sampler"""
    FREE = "FREE"
    SOUND = "SOUND"
    PATTERN = "PATTERN"
    BPM = "BPM"
    WRITE = "WRITE"
    RECORD = "RECORD"
    VOLUME = "VOLUME"
    TRIM = "TRIM"
    FX = "FX"


class FXSubMode(Enum):
    """Sub-modes for the FX workflow"""
    ONE_SHOT = "ONE_SHOT"
    PERMANENT = "PERMANENT"


@dataclass
class SliceData:
    """Represents a single slice in a kit"""
    start: float  # Start time in seconds
    duration: float  # Duration in seconds
    volume: float = 1.0  # Volume multiplier (0.0 to 1.0)
    pitch_offset: float = 0.0  # Pitch offset in semitones (-12.0 to +12.0)


@dataclass
class SoundSlot:
    """Represents a sound or kit in one of the 16 slots"""
    slot_number: int  # 1-16
    file_path: Optional[str] = None  # Path to the audio file
    is_kit: bool = False  # True for slots 9-16 (kits), False for 1-8 (oneshots)
    volume: float = 1.0  # Overall volume for this slot
    pitch_offset: float = 0.0  # Pitch offset in semitones for ALL sounds (-12.0 to +12.0)
    choke_group: int = 0  # Choke group (0=polyphonic, 1-15=monophonic group)
    slices: List[SliceData] = field(default_factory=lambda: [SliceData(0, 0) for _ in range(16)])

    @property
    def is_loaded(self) -> bool:
        """Check if this slot has a sound loaded"""
        return self.file_path is not None


@dataclass
class Pattern:
    """Represents a 16-step pattern with multi-track support"""
    pattern_number: int  # 1-16
    # Dictionary mapping slot_number -> list of 16 bools (steps)
    steps: Dict[int, List[bool]] = field(default_factory=dict)

    def toggle_step(self, slot_number: int, step: int):
        """Toggle a step for a given slot"""
        if slot_number not in self.steps:
            self.steps[slot_number] = [False] * 16
        self.steps[slot_number][step] = not self.steps[slot_number][step]

    def get_step(self, slot_number: int, step: int) -> bool:
        """Check if a step is active for a given slot"""
        if slot_number not in self.steps:
            return False
        return self.steps[slot_number][step]

    def clear(self):
        """Clear all steps in this pattern"""
        self.steps.clear()


class StateManager:
    """Main state manager for the BAPS-1 sampler"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.sounds_dir = os.path.join(data_dir, "sounds")
        self.kits_dir = os.path.join(data_dir, "kits")

        # Current mode
        self.mode = Mode.FREE

        # 16 sound/kit slots (1-indexed, so we'll use index 0-15 but display as 1-16)
        self.slots: List[SoundSlot] = [
            SoundSlot(slot_number=i+1, is_kit=(i >= 8))
            for i in range(16)
        ]

        # 16 patterns
        self.patterns: List[Pattern] = [
            Pattern(pattern_number=i+1)
            for i in range(16)
        ]

        # Pattern chain for playback
        self.pattern_chain: List[int] = []  # List of pattern numbers being edited
        self.playing_chain: List[int] = []  # Locked copy that's actually playing
        self.current_pattern_in_chain: int = 0  # Index in playing_chain

        # Currently selected items
        self.selected_slot: int = 1  # 1-16
        self.selected_pattern: int = 1  # 1-16
        self.last_played_pad: int = 1  # 1-16, tracks last pad pressed in FREE mode
        self.last_played_note: Optional[int] = None  # For oneshots, tracks last note/pitch

        # Playback state
        self.is_playing: bool = False
        self.current_step: int = 0  # 0-15

        # BPM and timing
        self.bpm: int = 120
        self.swing: float = 0.5  # 0.5 = straight, 0.75 = triplet swing

        # Master volume
        self.master_volume: float = 1.0

        # FX state
        self.fx_submode: FXSubMode = FXSubMode.ONE_SHOT
        self.fx_permanent_pad: int = 1  # Which pad/effect is selected in permanent FX mode
        self.fx_filter_high_pass: float = 0.0  # Normalized (0.0-1.0)
        self.fx_filter_low_pass: float = 0.0  # Normalized (0.0-1.0)
        self.fx_reverb_mix: float = 0.0
        self.fx_reverb_decay: float = 0.0
        self.fx_chorus_depth: float = 0.0
        self.fx_chorus_rate: float = 0.0

        # Available sound/kit files
        self.available_sounds: List[str] = []
        self.available_kits: List[str] = []

        # Scan for available files
        self._scan_audio_files()

    def _scan_audio_files(self):
        """Scan data directories for available audio files"""
        if os.path.exists(self.sounds_dir):
            self.available_sounds = sorted([
                f for f in os.listdir(self.sounds_dir)
                if f.endswith('.wav')
            ])

        if os.path.exists(self.kits_dir):
            self.available_kits = sorted([
                f for f in os.listdir(self.kits_dir)
                if f.endswith('.wav')
            ])

    def get_current_slot(self) -> SoundSlot:
        """Get the currently selected slot"""
        return self.slots[self.selected_slot - 1]

    def get_slot(self, slot_number: int) -> SoundSlot:
        """Get a specific slot by number (1-16)"""
        return self.slots[slot_number - 1]

    def get_current_pattern(self) -> Pattern:
        """Get the currently selected pattern"""
        return self.patterns[self.selected_pattern - 1]

    def get_pattern(self, pattern_number: int) -> Pattern:
        """Get a specific pattern by number (1-16)"""
        return self.patterns[pattern_number - 1]

    def load_sound_to_slot(self, slot_number: int, filename: str):
        """Load a sound file into a slot"""
        slot = self.get_slot(slot_number)

        if slot.is_kit:
            file_path = os.path.join(self.kits_dir, filename)
        else:
            file_path = os.path.join(self.sounds_dir, filename)

        slot.file_path = file_path

        if slot.is_kit:
            # Try to load metadata (auto-slice if missing)
            self._load_kit_metadata(slot, file_path)

    def _load_kit_metadata(self, slot: SoundSlot, wav_path: str):
        """Load kit slice metadata from JSON if it exists"""
        json_path = wav_path.replace('.wav', '.json')

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    slot.slices = [
                        SliceData(
                            start=s.get('start', 0),
                            duration=s.get('duration', 0),
                            volume=s.get('volume', 1.0),
                            pitch_offset=s.get('pitch_offset', 0.0)
                        )
                        for s in data.get('slices', [])
                    ]
                    # Ensure we have exactly 16 slices
                    while len(slot.slices) < 16:
                        slot.slices.append(SliceData(0, 0))
                    slot.slices = slot.slices[:16]
            except Exception as e:
                print(f"Error loading kit metadata: {e}")
                # Initialize with empty slices
                slot.slices = [SliceData(0, 0) for _ in range(16)]
        else:
            # No metadata file, perform automatic transient slicing
            self._auto_slice_kit(slot, wav_path)
            # Persist generated slices for faster reload next time
            self.save_kit_metadata(slot.slot_number)

    def _auto_slice_kit(self, slot: SoundSlot, wav_path: str):
        """Automatically chop a kit sample into slices based on transients."""
        try:
            audio, sample_rate = sf.read(wav_path)
        except Exception as e:
            print(f"Error auto-slicing kit {wav_path}: {e}")
            slot.slices = [SliceData(0, 0) for _ in range(16)]
            return

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if len(audio) == 0:
            slot.slices = [SliceData(0, 0) for _ in range(16)]
            return

        # Build amplitude envelope
        envelope = np.abs(audio)
        window = max(1, sample_rate // 400)  # ~2.5 ms smoothing
        kernel = np.ones(window) / window
        smoothed = np.convolve(envelope, kernel, mode='same')

        # Dynamic threshold
        mean = float(smoothed.mean())
        std = float(smoothed.std())
        threshold = max(0.02, mean + std * 0.6)
        min_gap = int(sample_rate * 0.02)  # 20ms between transients

        peaks = []
        last_idx = -min_gap
        for idx in range(1, len(smoothed) - 1):
            if smoothed[idx] >= threshold and smoothed[idx] >= smoothed[idx - 1] and smoothed[idx] > smoothed[idx + 1]:
                if idx - last_idx >= min_gap:
                    peaks.append(idx)
                    last_idx = idx
                    if len(peaks) >= 16:
                        break

        if not peaks:
            peaks = [0]
        else:
            peaks = [peaks[0]] + [p for i, p in enumerate(peaks[1:], start=1) if p - peaks[i - 1] >= min_gap]
            if peaks[0] > int(sample_rate * 0.01):
                peaks.insert(0, 0)

        peaks = sorted(set(peaks))
        if peaks[-1] != len(audio):
            peaks.append(len(audio))

        while len(peaks) <= 16:
            peaks.append(len(audio))

        slices: List[SliceData] = []
        for i in range(16):
            start_sample = peaks[i]
            end_sample = peaks[i + 1]
            if end_sample <= start_sample:
                duration = 0.0
            else:
                duration = (end_sample - start_sample) / sample_rate

            slices.append(
                SliceData(
                    start=start_sample / sample_rate,
                    duration=duration,
                    volume=1.0,
                    pitch_offset=0.0
                )
            )

        slot.slices = slices

    def save_kit_metadata(self, slot_number: int):
        """Save kit slice metadata to JSON"""
        slot = self.get_slot(slot_number)

        if not slot.is_kit or not slot.file_path:
            return

        json_path = slot.file_path.replace('.wav', '.json')

        data = {
            'slices': [
                {
                    'start': s.start,
                    'duration': s.duration,
                    'volume': s.volume,
                    'pitch_offset': s.pitch_offset
                }
                for s in slot.slices
            ]
        }

        try:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving kit metadata: {e}")

    def add_to_pattern_chain(self, pattern_number: int):
        """Add a pattern to the chain"""
        self.pattern_chain.append(pattern_number)

    def clear_pattern_chain(self):
        """Clear the pattern chain"""
        self.pattern_chain.clear()
        self.current_pattern_in_chain = 0

    def lock_playing_chain(self):
        """Lock the current chain for playback (called when play is pressed)"""
        if self.pattern_chain:
            self.playing_chain = self.pattern_chain.copy()
        else:
            self.playing_chain = []
        self.current_pattern_in_chain = 0

    def get_active_pattern_number(self) -> Optional[int]:
        """Get the currently playing pattern number"""
        if not self.playing_chain:
            return self.selected_pattern
        return self.playing_chain[self.current_pattern_in_chain]

    def advance_pattern_chain(self):
        """Move to the next pattern in the chain, update cache if chain completed"""
        if not self.playing_chain:
            return

        next_index = (self.current_pattern_in_chain + 1) % len(self.playing_chain)

        # Check if we're wrapping back to 0 (chain completed)
        if next_index == 0:
            # Update playing_chain with current pattern_chain (queue swap)
            if self.pattern_chain:
                self.playing_chain = self.pattern_chain.copy()
            else:
                # No chain set, so just loop selected pattern
                self.playing_chain = []

        self.current_pattern_in_chain = next_index

    def save_state(self, filepath: str = "state.json"):
        """Save current state to JSON"""
        state_data = {
            'bpm': self.bpm,
            'swing': self.swing,
            'master_volume': self.master_volume,
            'selected_slot': self.selected_slot,
            'selected_pattern': self.selected_pattern,
            'pattern_chain': self.pattern_chain,
            'fx': {
                'permanent_pad': self.fx_permanent_pad,
                'filter_high_pass': self.fx_filter_high_pass,
                'filter_low_pass': self.fx_filter_low_pass,
                'reverb_mix': self.fx_reverb_mix,
                'reverb_decay': self.fx_reverb_decay,
                'chorus_depth': self.fx_chorus_depth,
                'chorus_rate': self.fx_chorus_rate,
            },
            'slots': [
                {
                    'slot_number': slot.slot_number,
                    'file_path': slot.file_path,
                    'volume': slot.volume,
                    'pitch_offset': slot.pitch_offset,
                    'choke_group': slot.choke_group,
                    'is_kit': slot.is_kit
                }
                for slot in self.slots
            ],
            'patterns': [
                {
                    'pattern_number': p.pattern_number,
                    'steps': {str(k): v for k, v in p.steps.items()}  # Convert int keys to str for JSON
                }
                for p in self.patterns
            ]
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            print(f"State saved to {filepath}")
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self, filepath: str = "state.json"):
        """Load state from JSON"""
        if not os.path.exists(filepath):
            print(f"State file {filepath} not found")
            return

        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)

            self.bpm = state_data.get('bpm', 120)
            self.swing = state_data.get('swing', 0.5)
            self.master_volume = state_data.get('master_volume', 1.0)
            self.selected_slot = state_data.get('selected_slot', 1)
            self.selected_pattern = state_data.get('selected_pattern', 1)
            self.pattern_chain = state_data.get('pattern_chain', [])
            fx_state = state_data.get('fx', {})
            self.fx_permanent_pad = max(1, min(16, int(fx_state.get('permanent_pad', 1))))
            self.fx_filter_high_pass = max(0.0, min(1.0, float(fx_state.get('filter_high_pass', 0.0))))
            self.fx_filter_low_pass = max(0.0, min(1.0, float(fx_state.get('filter_low_pass', 0.0))))
            self.fx_reverb_mix = max(0.0, min(1.0, float(fx_state.get('reverb_mix', 0.0))))
            self.fx_reverb_decay = max(0.0, min(1.0, float(fx_state.get('reverb_decay', 0.0))))
            self.fx_chorus_depth = max(0.0, min(1.0, float(fx_state.get('chorus_depth', 0.0))))
            self.fx_chorus_rate = max(0.0, min(1.0, float(fx_state.get('chorus_rate', 0.0))))

            # Load slots
            for slot_data in state_data.get('slots', []):
                slot_num = slot_data['slot_number']
                slot = self.get_slot(slot_num)
                slot.volume = slot_data.get('volume', 1.0)
                slot.pitch_offset = slot_data.get('pitch_offset', 0.0)
                slot.choke_group = slot_data.get('choke_group', 0)

                if slot_data.get('file_path'):
                    # Extract just the filename
                    filename = os.path.basename(slot_data['file_path'])
                    self.load_sound_to_slot(slot_num, filename)

            # Load patterns
            for pattern_data in state_data.get('patterns', []):
                pattern_num = pattern_data['pattern_number']
                pattern = self.get_pattern(pattern_num)
                # Convert str keys back to int
                pattern.steps = {int(k): v for k, v in pattern_data.get('steps', {}).items()}

            print(f"State loaded from {filepath}")
        except Exception as e:
            print(f"Error loading state: {e}")
