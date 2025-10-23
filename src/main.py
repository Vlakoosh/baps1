"""
BAPS-1 Portable Sampler
Main application entry point
"""

import sys
import time
import threading
from state_manager import StateManager, Mode, FXSubMode
from audio_engine import AudioEngine
from display_manager import DisplayManager
from input_handler import InputHandler, Button
from sequencer import Sequencer

ONE_SHOT_EFFECT_NAMES = {
    1: "Beat Repeat",
    2: "Vinyl Stop",
}

PERMANENT_EFFECTS = {
    1: {"name": "Filter", "hint": "K1 HP  K2 LP"},
    2: {"name": "Reverb", "hint": "K1 MIX  K2 DECAY"},
    3: {"name": "Chorus", "hint": "K1 DEPTH  K2 RATE"},
}


class BAPS1:
    """Main application class for BAPS-1 sampler"""

    def __init__(self):
        print("Initializing BAPS-1 Sampler...")

        # Initialize components
        self.state = StateManager(data_dir="../data")
        self.audio = AudioEngine(sample_rate=22050)
        self.display = DisplayManager(use_emulator=False)
        self.input = InputHandler()
        self.sequencer = Sequencer(bpm=self.state.bpm, swing=self.state.swing)

        # Set up input callbacks
        self.input.on_button_press = self.handle_button_press
        self.input.on_button_release = self.handle_button_release
        self.input.on_button_hold = self.handle_button_hold
        self.input.on_encoder_turn = self.handle_encoder_turn

        # Set up sequencer callback
        self.sequencer.on_step = self.handle_sequencer_step

        # FX runtime state
        self.fx_lock = threading.Lock()
        self.fx_one_shot_held = set()
        self.fx_one_shot_pending = set()
        self.fx_one_shot_active = set()

        # Track chain building state (for auto-clear on first pad)
        self.chain_prev_length = 0

        # Start audio engine and sequencer
        self.audio.start()
        self.sequencer.start()

        # Load saved state if it exists
        self.state.load_state("../state.json")

        # Sync sequencer with loaded state
        self.sequencer.set_bpm(self.state.bpm)
        self.sequencer.set_swing(self.state.swing)

        # Apply any saved FX settings to the audio engine
        self._update_permanent_fx_audio()

        # Preload any already-assigned sounds
        self._preload_assigned_sounds()

        print("BAPS-1 Ready!")
        print("\nControls:")
        print("  Pads: 1234 QWER ASDF ZXCV")
        print("  Modes: F1=SOUND F2=PATTERN F3=BPM F4=WRITE F5=RECORD F6=VOLUME F7=TRIM")
        print("  Play: SPACE")
        print("  Knobs: Arrow keys (L/R for knob1, U/D for knob2)")
        print("  Special: Hold F1+SPACE=Hard Reset, Hold F3+SPACE=Backup")
        print("  Quit: ESC or close window\n")

    def _preload_assigned_sounds(self):
        """Preload any sounds that are already assigned to slots"""
        filepaths = []
        for slot in self.state.slots:
            if slot.is_loaded:
                filepaths.append(slot.file_path)

        if filepaths:
            print(f"Preloading {len(filepaths)} sounds...")
            self.audio.preload_sounds(filepaths)

    def _preload_pattern_sounds(self):
        """Preload all sounds used in patterns about to be played"""
        # Determine which patterns will be played (from playing_chain)
        if self.state.playing_chain:
            pattern_numbers = self.state.playing_chain
        else:
            pattern_numbers = [self.state.selected_pattern]

        # Collect all unique slot numbers from all patterns
        slot_numbers = set()
        for pattern_num in pattern_numbers:
            pattern = self.state.get_pattern(pattern_num)
            for track_id in pattern.steps.keys():
                # Decode track_id to get slot number
                # track_id = (slot_number - 1) * 16 + pad_number
                slot_num = (track_id - 1) // 16 + 1
                slot_numbers.add(slot_num)

        # Collect file paths for all these slots
        filepaths = []
        for slot_num in slot_numbers:
            slot = self.state.get_slot(slot_num)
            if slot.is_loaded:
                filepaths.append(slot.file_path)

        # Preload all sounds
        if filepaths:
            print(f"Preloading {len(filepaths)} pattern sounds...")
            self.audio.preload_sounds(filepaths)
            print("Preloading complete!")

    def _update_permanent_fx_audio(self):
        """Sync permanent FX settings to the audio engine."""
        pad = self.state.fx_permanent_pad

        if pad == 1:
            self.audio.set_filter_params(
                self.state.fx_filter_high_pass,
                self.state.fx_filter_low_pass
            )
        elif pad == 2:
            self.audio.set_reverb_params(
                self.state.fx_reverb_mix,
                self.state.fx_reverb_decay
            )
        elif pad == 3:
            self.audio.set_chorus_params(
                self.state.fx_chorus_depth,
                self.state.fx_chorus_rate
            )
        else:
            self.audio.set_effect_none()

    def create_backup(self):
        """Create a timestamped backup of the current project"""
        import os
        import shutil
        from datetime import datetime

        print("\n=== CREATING BACKUP ===")

        # Create backups directory if it doesn't exist
        backup_dir = "../backups"
        os.makedirs(backup_dir, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)
        os.makedirs(backup_path, exist_ok=True)

        # Save current state
        self.state.save_state("../state.json")

        # Copy state.json
        shutil.copy2("../state.json", os.path.join(backup_path, "state.json"))
        print(f"[OK] Backed up state.json")

        # Copy all kit metadata files
        data_dir = "../data"
        metadata_count = 0
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".json"):
                    shutil.copy2(
                        os.path.join(data_dir, file),
                        os.path.join(backup_path, file)
                    )
                    metadata_count += 1

        if metadata_count > 0:
            print(f"[OK] Backed up {metadata_count} kit metadata files")

        print(f"[OK] Backup complete: {backup_name}")
        print(f"     Location: {backup_path}")

    def handle_button_press(self, button: Button):
        """Handle button press events"""
        # Check if it's a mode button - toggle behavior
        if button == Button.SOUND:
            if self.state.mode == Mode.SOUND:
                self.state.mode = Mode.FREE
            else:
                self.state.mode = Mode.SOUND
            self.update_display()

        elif button == Button.PATTERN:
            # Just toggle mode, don't clear chain
            if self.state.mode == Mode.PATTERN:
                self.state.mode = Mode.FREE
                # Reset chain building tracking when exiting PATTERN mode
                self.chain_prev_length = 0
            else:
                self.state.mode = Mode.PATTERN
            self.update_display()

        elif button == Button.BPM:
            if self.state.mode == Mode.BPM:
                self.state.mode = Mode.FREE
            else:
                self.state.mode = Mode.BPM
            self.update_display()

        elif button == Button.WRITE:
            if self.state.mode == Mode.WRITE:
                self.state.mode = Mode.FREE
            else:
                self.state.mode = Mode.WRITE
            self.update_display()

        elif button == Button.VOLUME:
            if self.state.mode == Mode.VOLUME:
                self.state.mode = Mode.FREE
            else:
                self.state.mode = Mode.VOLUME
            self.update_display()

        elif button == Button.TRIM:
            if self.state.mode == Mode.TRIM:
                self.state.mode = Mode.FREE
            else:
                self.state.mode = Mode.TRIM
            self.update_display()

        elif button == Button.RECORD:
            if self.state.mode == Mode.RECORD:
                self.state.mode = Mode.FREE
            else:
                self.state.mode = Mode.RECORD
            self.update_display()

        elif button == Button.FX:
            if self.state.mode != Mode.FX:
                self.state.mode = Mode.FX
                self.state.fx_submode = FXSubMode.ONE_SHOT
                with self.fx_lock:
                    self.fx_one_shot_held.clear()
                    self.fx_one_shot_pending.clear()
                    self.fx_one_shot_active.clear()
            else:
                if self.state.fx_submode == FXSubMode.ONE_SHOT:
                    self.state.fx_submode = FXSubMode.PERMANENT
                    with self.fx_lock:
                        self.fx_one_shot_held.clear()
                        self.fx_one_shot_pending.clear()
                        self.fx_one_shot_active.clear()
                else:
                    self.state.mode = Mode.FREE
                    self.state.fx_submode = FXSubMode.ONE_SHOT
                    with self.fx_lock:
                        self.fx_one_shot_held.clear()
                        self.fx_one_shot_pending.clear()
                        self.fx_one_shot_active.clear()
            self.update_display()

        elif button == Button.PLAY:
            # Check for backup shortcut (Hold BPM + Press PLAY)
            if self.input.is_button_held(Button.BPM):
                self.create_backup()
            else:
                self.toggle_playback()

        # Handle pad presses based on current mode
        elif self.input.is_pad_button(button):
            pad_num = self.input.get_pad_number(button)
            self.handle_pad_press(pad_num)

    def handle_button_release(self, button: Button):
        """Handle button release events"""
        # No longer needed for mode buttons (using toggle instead)
        if self.state.mode == Mode.FX and self.input.is_pad_button(button):
            pad_num = self.input.get_pad_number(button)
            if pad_num is None:
                return

            if self.state.fx_submode == FXSubMode.ONE_SHOT:
                with self.fx_lock:
                    self.fx_one_shot_held.discard(pad_num)
                    self.fx_one_shot_pending.discard(pad_num)
                    self.fx_one_shot_active.discard(pad_num)

            self.update_display()

    def handle_button_hold(self, button: Button):
        """Handle button hold events"""
        # Holding SOUND button + PLAY = hard reset
        if button == Button.PLAY and self.input.is_button_held(Button.SOUND):
            print("\n=== HARD RESET ===")
            print("Clearing all patterns and resetting to defaults...")

            # Clear all patterns
            for pattern in self.state.patterns:
                pattern.clear()

            # Reset state
            self.state.pattern_chain.clear()
            self.state.bpm = 120
            self.state.swing = 0.5
            self.state.selected_pattern = 1
            self.state.last_played_pad = 1

            # Stop playback
            if self.state.is_playing:
                self.toggle_playback()

            self.sequencer.set_bpm(120)
            self.sequencer.set_swing(0.5)

            print("Reset complete!")
            self.update_display()

    def handle_encoder_turn(self, encoder_num: int, direction: int):
        """Handle encoder turn events"""
        if self.state.mode == Mode.FREE:
            # Pitch tuning in FREE mode
            slot = self.state.get_slot(self.state.selected_slot)

            if encoder_num == 1:
                # Knob 1: ALL pitch tuning
                slot.pitch_offset = max(-12.0, min(12.0, slot.pitch_offset + direction * 0.25))
            elif encoder_num == 2 and slot.is_kit:
                # Knob 2: Individual pad tuning (kits only)
                slice_idx = self.state.last_played_pad - 1
                slice_data = slot.slices[slice_idx]
                slice_data.pitch_offset = max(-12.0, min(12.0, slice_data.pitch_offset + direction * 0.25))
                # Save kit metadata
                self.state.save_kit_metadata(self.state.selected_slot)

            self.update_display()

        elif self.state.mode == Mode.BPM:
            if encoder_num == 2:
                # Knob 2: BPM adjustment
                self.state.bpm = max(60, min(300, self.state.bpm + direction))
                self.sequencer.set_bpm(self.state.bpm)
            elif encoder_num == 1:
                # Knob 1: Swing adjustment (5% increments)
                self.state.swing = max(0.0, min(1.0, self.state.swing + direction * 0.05))
                self.sequencer.set_swing(self.state.swing)
            self.update_display()

        elif self.state.mode == Mode.SOUND:
            if encoder_num == 1:
                # Knob 1: Navigate through available sounds
                self.navigate_sound_list(direction)
            elif encoder_num == 2:
                # Knob 2: Adjust choke group (0-15)
                slot = self.state.get_current_slot()
                slot.choke_group = max(0, min(15, slot.choke_group + direction))
                print(f"Choke group set to {slot.choke_group} (0=poly, 1-15=choke groups)")
            self.update_display()

        elif self.state.mode == Mode.VOLUME:
            if encoder_num == 1:
                # Knob 1: Master volume
                self.state.master_volume = max(0.0, min(1.0, self.state.master_volume + direction * 0.01))
                self.audio.master_volume = self.state.master_volume
            elif encoder_num == 2:
                # Knob 2: Selected slot/channel volume
                slot = self.state.get_slot(self.state.selected_slot)
                slot.volume = max(0.0, min(1.0, slot.volume + direction * 0.01))
                # Preview the sound
                if slot.is_loaded:
                    self.play_pad(self.state.selected_slot)
            self.update_display()

        elif self.state.mode == Mode.TRIM:
            slot = self.state.get_current_slot()
            if not slot.is_loaded:
                return

            if slot.is_kit:
                # Adjust trim for last played pad
                slice_idx = self.state.last_played_pad - 1
                slice_data = slot.slices[slice_idx]
                duration = self.audio.get_audio_duration(slot.file_path)

                if encoder_num == 1:
                    # Knob 1: Adjust start time
                    slice_data.start = max(0, min(duration - slice_data.duration,
                                                  slice_data.start + direction * 0.01))
                elif encoder_num == 2:
                    # Knob 2: Adjust duration
                    max_duration = duration - slice_data.start
                    slice_data.duration = max(0.001, min(max_duration,
                                                         slice_data.duration + direction * 0.01))

                # Preview on each adjustment
                self.play_pad(self.state.last_played_pad)
                self.state.save_kit_metadata(self.state.selected_slot)
            else:
                # For oneshots, we could implement trim here too
                # For now, just do nothing
                pass

            self.update_display()

        elif self.state.mode == Mode.FX:
            if self.state.fx_submode == FXSubMode.PERMANENT:
                pad = self.state.fx_permanent_pad
                changed = False

                if pad == 1:
                    if encoder_num == 1:
                        new_hp = max(0.0, min(1.0, self.state.fx_filter_high_pass + direction * 0.02))
                        if new_hp != self.state.fx_filter_high_pass:
                            self.state.fx_filter_high_pass = new_hp
                            changed = True
                    elif encoder_num == 2:
                        new_lp = max(0.0, min(1.0, self.state.fx_filter_low_pass + direction * 0.02))
                        if new_lp != self.state.fx_filter_low_pass:
                            self.state.fx_filter_low_pass = new_lp
                            changed = True

                elif pad == 2:
                    if encoder_num == 1:
                        new_mix = max(0.0, min(1.0, self.state.fx_reverb_mix + direction * 0.02))
                        if new_mix != self.state.fx_reverb_mix:
                            self.state.fx_reverb_mix = new_mix
                            changed = True
                    elif encoder_num == 2:
                        new_decay = max(0.0, min(1.0, self.state.fx_reverb_decay + direction * 0.02))
                        if new_decay != self.state.fx_reverb_decay:
                            self.state.fx_reverb_decay = new_decay
                            changed = True

                elif pad == 3:
                    if encoder_num == 1:
                        new_depth = max(0.0, min(1.0, self.state.fx_chorus_depth + direction * 0.02))
                        if new_depth != self.state.fx_chorus_depth:
                            self.state.fx_chorus_depth = new_depth
                            changed = True
                    elif encoder_num == 2:
                        new_rate = max(0.0, min(1.0, self.state.fx_chorus_rate + direction * 0.02))
                        if new_rate != self.state.fx_chorus_rate:
                            self.state.fx_chorus_rate = new_rate
                            changed = True

                if changed:
                    self._update_permanent_fx_audio()
                    self.update_display()

    def handle_pad_press(self, pad_num: int):
        """Handle pad press in different modes"""
        if self.state.mode == Mode.FREE or self.state.mode == Mode.VOLUME:
            # Play the sound and remember which pad was played
            self.state.last_played_pad = pad_num
            self.play_pad(pad_num)

        elif self.state.mode == Mode.SOUND:
            # Select this slot for sound loading
            self.state.selected_slot = pad_num
            # Reset last played pad when switching slots
            # This ensures WRITE mode starts fresh for the new slot
            self.state.last_played_pad = 1
            self.update_display()

        elif self.state.mode == Mode.PATTERN:
            # Check if PATTERN button is being held
            if self.input.is_button_held(Button.PATTERN):
                # Auto-clear on first pad press while holding
                if len(self.state.pattern_chain) > 0 and self.chain_prev_length == 0:
                    self.state.clear_pattern_chain()
                    print("Pattern chain cleared - building new chain")

                # Add to chain while holding PATTERN button
                self.state.add_to_pattern_chain(pad_num)
                print(f"Added pattern {pad_num} to chain (total: {len(self.state.pattern_chain)})")

                # Track that we're building a chain
                self.chain_prev_length = len(self.state.pattern_chain)

                # BUG FIX: If playback is active and playing_chain is empty,
                # immediately lock the new chain so it starts playing
                if self.state.is_playing and not self.state.playing_chain:
                    self.state.lock_playing_chain()
                    print(f"New chain queued during playback: {self.state.playing_chain}")
            else:
                # Just select pattern
                self.state.selected_pattern = pad_num
                # Reset chain building tracking when not holding PATTERN
                self.chain_prev_length = 0
            self.update_display()

        elif self.state.mode == Mode.WRITE:
            # Toggle step in pattern
            pattern = self.state.get_current_pattern()

            # Encode both slot and pad into track_id
            # track_id = (slot_number - 1) * 16 + pad_number
            # This ensures each slot+pad combination is unique
            slot_num = self.state.selected_slot
            pad = self.state.last_played_pad
            track_id = (slot_num - 1) * 16 + pad

            pattern.toggle_step(track_id, pad_num - 1)
            self.update_display()

        elif self.state.mode == Mode.TRIM:
            # Select pad for trimming and remember it
            self.state.last_played_pad = pad_num
            self.play_pad(pad_num)

        elif self.state.mode == Mode.FX:
            self.state.last_played_pad = pad_num
            if self.state.fx_submode == FXSubMode.ONE_SHOT:
                if pad_num in ONE_SHOT_EFFECT_NAMES:
                    with self.fx_lock:
                        self.fx_one_shot_held.add(pad_num)
                        self.fx_one_shot_pending.add(pad_num)
            else:
                self.state.fx_permanent_pad = pad_num
                self._update_permanent_fx_audio()

        self.update_display()

    def play_pad(self, pad_num: int):
        """Play a sound from a pad"""
        # For oneshots, use the selected slot
        # For kits, use the pad number as the slot
        slot = self.state.get_slot(self.state.selected_slot)

        if not slot.is_loaded:
            return

        if slot.is_kit:
            # Play the appropriate slice from the kit
            slice_idx = pad_num - 1
            slice_data = slot.slices[slice_idx]

            if slice_data.duration > 0:
                # Calculate total pitch: ALL + PAD
                total_pitch = slot.pitch_offset + slice_data.pitch_offset

                # Use choke group for voice management
                self.audio.play_slice(
                    slot.file_path,
                    slice_data.start,
                    slice_data.duration,
                    volume=slot.volume * slice_data.volume,
                    slot_id=None,  # Legacy parameter
                    pitch_shift=total_pitch,
                    choke_group=slot.choke_group
                )
        else:
            # Play as pitched oneshot
            # Each pad is +1 semitone from the previous
            # Pad 1 = root (0), Pad 2 = +1, Pad 3 = +2, etc.
            keyboard_pitch = pad_num - 1
            # Add the ALL pitch offset
            total_pitch = keyboard_pitch + slot.pitch_offset

            # Use choke group for voice management
            self.audio.play_sound(
                slot.file_path,
                volume=slot.volume,
                pitch_shift=total_pitch,
                slot_id=self.state.selected_slot,  # Legacy parameter
                choke_group=slot.choke_group
            )
            self.state.last_played_note = keyboard_pitch

    def navigate_sound_list(self, direction: int):
        """Navigate through available sounds in SOUND mode"""
        slot = self.state.get_current_slot()

        # Get the appropriate file list with EMPTY option
        base_list = self.state.available_kits if slot.is_kit else self.state.available_sounds
        file_list = ["[EMPTY]"] + base_list

        if not file_list:
            return

        # Find current file index
        current_idx = 0  # Default to EMPTY
        if slot.is_loaded:
            import os
            current_file = os.path.basename(slot.file_path)
            if current_file in file_list:
                current_idx = file_list.index(current_file)

        # Navigate
        new_idx = (current_idx + direction) % len(file_list)
        new_file = file_list[new_idx]

        # Handle EMPTY selection
        if new_file == "[EMPTY]":
            # Clear the slot
            slot.file_path = None
            print(f"Cleared slot {self.state.selected_slot}")
        else:
            # Load the new sound
            self.state.load_sound_to_slot(self.state.selected_slot, new_file)
            self.audio.load_audio(slot.file_path)

        # Reset last played pad when changing sounds
        # This ensures WRITE mode starts fresh for the new sound
        self.state.last_played_pad = 1

        self.update_display()

    def handle_sequencer_step(self, step: int):
        """
        Handle a sequencer step - trigger sounds based on pattern

        Args:
            step: The current step (0-15)
        """
        # Determine if we just hit the start of a new pattern loop
        is_loop_start = (step == 0)
        if is_loop_start:
            # Activate any one-shot effects that were armed for this loop
            with self.fx_lock:
                self.fx_one_shot_active = set(self.fx_one_shot_pending)
                # Re-arm held pads so the effect retriggers on subsequent loops
                if self.fx_one_shot_held:
                    self.fx_one_shot_pending = set(self.fx_one_shot_held)
                else:
                    self.fx_one_shot_pending.clear()
        with self.fx_lock:
            active_one_shots = set(self.fx_one_shot_active)

        if is_loop_start:
            beat_samples = max(1, int(self.audio.sample_rate * 60.0 / max(1, self.state.bpm)))
            pattern_samples = beat_samples * 4
            self.audio.start_pattern_cycle(pattern_samples, beat_samples, active_one_shots)

        # Check if we just wrapped around to step 0 (pattern completed)
        pattern_completed = (step == 0 and self.state.current_step == 15)

        # Update current step
        self.state.current_step = step

        # Get the active pattern (from playing_chain or selected)
        if self.state.playing_chain:
            # If pattern just completed, advance to next in chain
            if pattern_completed:
                self.state.advance_pattern_chain()

            # Re-check if playing_chain is still valid after advancing
            # (it might become empty if user cleared the chain)
            if self.state.playing_chain and self.state.current_pattern_in_chain < len(self.state.playing_chain):
                pattern_idx = self.state.current_pattern_in_chain
                pattern_num = self.state.playing_chain[pattern_idx]
                pattern = self.state.get_pattern(pattern_num)
            else:
                # Chain became empty, fall back to selected pattern
                pattern = self.state.get_current_pattern()
        else:
            # Use the selected pattern
            pattern = self.state.get_current_pattern()

        # Check each track to see if it should trigger on this step
        for track_id, steps in pattern.steps.items():
            if steps[step]:
                # Decode track_id to get slot and pad
                # track_id = (slot_number - 1) * 16 + pad_number
                slot_num = (track_id - 1) // 16 + 1
                pad_num = (track_id - 1) % 16 + 1

                # Get the slot
                slot = self.state.get_slot(slot_num)

                if not slot.is_loaded:
                    continue

                if slot.is_kit:
                    # Play kit slice
                    slice_idx = pad_num - 1
                    if 0 <= slice_idx < len(slot.slices):
                        slice_data = slot.slices[slice_idx]
                        if slice_data.duration > 0:
                            total_pitch = slot.pitch_offset + slice_data.pitch_offset
                            self.audio.play_slice(
                                slot.file_path,
                                slice_data.start,
                                slice_data.duration,
                                volume=slot.volume * slice_data.volume,
                                slot_id=None,  # Legacy parameter
                                pitch_shift=total_pitch,
                                choke_group=slot.choke_group
                            )
                else:
                    # Play oneshot at the programmed pad's pitch
                    keyboard_pitch = pad_num - 1
                    total_pitch = keyboard_pitch + slot.pitch_offset
                    # Use choke group for voice management
                    self.audio.play_sound(
                        slot.file_path,
                        volume=slot.volume,
                        pitch_shift=total_pitch,
                        slot_id=slot_num,  # Legacy parameter
                        choke_group=slot.choke_group
                    )

    def toggle_playback(self):
        """Toggle pattern playback"""
        self.state.is_playing = not self.state.is_playing

        if self.state.is_playing:
            print("Playback started")

            # Lock the current chain for playback (queue system)
            self.state.lock_playing_chain()

            if self.state.playing_chain:
                print(f"Playing pattern chain: {self.state.playing_chain}")
            else:
                print(f"Playing pattern {self.state.selected_pattern}")

            # Preload all audio samples before starting playback
            # This prevents first-loop timing issues from loading delays
            self._preload_pattern_sounds()

            self.sequencer.play()
        else:
            print("Playback stopped")
            self.sequencer.pause()
            self.audio.stop_all()

        self.update_display()

    def update_display(self):
        """Update the display based on current state"""
        if self.state.mode == Mode.FREE:
            level = self.audio.get_current_level()  # Real-time audio level
            slot = self.state.get_slot(self.state.selected_slot)

            # Get pitch values
            pitch_all = slot.pitch_offset
            pitch_pad = 0.0
            if slot.is_kit and self.state.last_played_pad <= 16:
                slice_idx = self.state.last_played_pad - 1
                pitch_pad = slot.slices[slice_idx].pitch_offset

            self.display.render_free_mode(
                pad_number=self.state.last_played_pad,
                level=level,
                is_playing=self.state.is_playing,
                pitch_all=pitch_all,
                pitch_pad=pitch_pad,
                is_kit=slot.is_kit,
                current_step=self.state.current_step
            )

        elif self.state.mode == Mode.SOUND:
            slot = self.state.get_current_slot()
            base_list = self.state.available_kits if slot.is_kit else self.state.available_sounds

            # Add EMPTY as first option
            file_list = ["[EMPTY]"] + base_list

            filename = "[EMPTY]"
            file_idx = 0
            if slot.is_loaded:
                import os
                filename = os.path.basename(slot.file_path)
                if filename in file_list:
                    file_idx = file_list.index(filename)
                else:
                    file_idx = 0

            self.display.render_sound_mode(
                pad_number=self.state.selected_slot,
                filename=filename,
                file_index=file_idx,
                file_list=file_list,
                choke_group=slot.choke_group
            )

        elif self.state.mode == Mode.BPM:
            self.display.render_bpm_mode(
                pad_number=self.state.last_played_pad,
                bpm=self.state.bpm,
                swing=self.state.swing
            )

        elif self.state.mode == Mode.PATTERN:
            self.display.render_pattern_mode(
                pad_number=self.state.selected_pattern,
                pattern_number=self.state.selected_pattern,
                chain_length=len(self.state.pattern_chain)
            )

        elif self.state.mode == Mode.WRITE:
            # Get which steps are active for the current track
            pattern = self.state.get_current_pattern()

            # Encode both slot and pad into track_id
            slot_num = self.state.selected_slot
            pad = self.state.last_played_pad
            track_id = (slot_num - 1) * 16 + pad

            steps_active = [False] * 16
            if track_id in pattern.steps:
                steps_active = pattern.steps[track_id].copy()

            self.display.render_write_mode(
                pad_number=pad,  # Show which pad is being programmed
                pattern_number=self.state.selected_pattern,
                current_step=self.state.current_step,
                steps_active=steps_active
            )

        elif self.state.mode == Mode.VOLUME:
            slot = self.state.get_slot(self.state.selected_slot)

            self.display.render_volume_mode(
                pad_number=self.state.selected_slot,
                pad_volume=slot.volume,
                kit_volume=self.state.master_volume
            )

        elif self.state.mode == Mode.TRIM:
            slot = self.state.get_current_slot()
            if slot.is_loaded and slot.is_kit:
                slice_idx = self.state.last_played_pad - 1
                slice_data = slot.slices[slice_idx]
                total_duration = self.audio.get_audio_duration(slot.file_path)

                # Get audio data for waveform visualization
                audio_data = self.audio.load_audio(slot.file_path)

                self.display.render_trim_mode(
                    pad_number=self.state.last_played_pad,
                    start_time=slice_data.start,
                    duration=slice_data.duration,
                    total_duration=total_duration,
                    audio_data=audio_data
                )

        elif self.state.mode == Mode.FX:
            pad = self.state.fx_permanent_pad
            effect_cfg = PERMANENT_EFFECTS.get(pad)
            effect_name = effect_cfg["name"] if effect_cfg else "Empty"
            knob_hint = effect_cfg["hint"] if effect_cfg else ""

            knob_left_value = 0.0
            knob_right_value = 0.0
            knob_left_text = "--"
            knob_right_text = "--"

            if pad == 1:
                knob_left_value = self.state.fx_filter_high_pass
                knob_right_value = self.state.fx_filter_low_pass
                knob_left_text = f"{int(knob_left_value * 100):02d}%"
                knob_right_text = f"{int(knob_right_value * 100):02d}%"
            elif pad == 2:
                knob_left_value = self.state.fx_reverb_mix
                knob_right_value = self.state.fx_reverb_decay
                knob_left_text = f"{int(knob_left_value * 100):02d}%"
                knob_right_text = f"{int(knob_right_value * 100):02d}%"
            elif pad == 3:
                knob_left_value = self.state.fx_chorus_depth
                knob_right_value = self.state.fx_chorus_rate
                knob_left_text = f"{int(knob_left_value * 100):02d}%"
                rate_hz = 0.1 + knob_right_value * 4.9
                knob_right_text = f"{rate_hz:3.1f}Hz"

            oneshot_hint = ""
            if ONE_SHOT_EFFECT_NAMES:
                parts = []
                for pad_id in sorted(ONE_SHOT_EFFECT_NAMES.keys()):
                    name = ONE_SHOT_EFFECT_NAMES[pad_id]
                    abbrev = "".join(word[0] for word in name.split()).upper()[:3]
                    parts.append(f"P{pad_id:02d}:{abbrev}")
                oneshot_hint = " ".join(parts[:3])
            with self.fx_lock:
                active_pads = sorted(self.fx_one_shot_active)
                held_pads = sorted(self.fx_one_shot_held)
                pending_pads = sorted(self.fx_one_shot_pending)

            self.display.render_fx_mode(
                submode=self.state.fx_submode.value,
                pad_number=self.state.last_played_pad,
                is_playing=self.state.is_playing,
                active_pads=active_pads,
                held_pads=held_pads,
                pending_pads=pending_pads,
                oneshot_hint=oneshot_hint,
                permanent_pad=self.state.fx_permanent_pad,
                effect_name=effect_name,
                knob_left_value=knob_left_value,
                knob_right_value=knob_right_value,
                knob_left_text=knob_left_text,
                knob_right_text=knob_right_text,
                knob_hint=knob_hint
            )

        elif self.state.mode == Mode.RECORD:
            self.display.render_record_mode(
                pad_number=self.state.last_played_pad,
                is_recording=False
            )

    def run(self):
        """Main application loop"""
        running = True
        last_display_update = 0
        display_update_interval = 0.016  # Update display at ~60 Hz for smooth animations

        try:
            # Initial display update
            self.update_display()

            while running:
                # Poll input events
                if not self.input.poll_events():
                    running = False
                    break

                # Update display periodically
                current_time = time.time()
                if current_time - last_display_update >= display_update_interval:
                    self.update_display()
                    last_display_update = current_time

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")

        # Save state
        self.state.save_state("../state.json")

        # Stop sequencer
        self.sequencer.stop()

        # Stop audio
        self.audio.stop()

        # Clean up display
        self.display.cleanup()

        # Clean up input
        self.input.cleanup()

        print("Goodbye!")


def main():
    """Main entry point"""
    app = BAPS1()
    app.run()


if __name__ == "__main__":
    main()
