"""
Sequencer for BAPS-1 Sampler
Handles pattern playback with BPM timing and swing
"""

import threading
import time
from typing import Callable, Optional


class Sequencer:
    """Handles pattern playback timing and step sequencing"""

    def __init__(self, bpm: int = 120, swing: float = 0.5):
        self.bpm = bpm
        self.swing = swing  # 0.5 = straight, 0.75 = triplet swing

        self.is_playing = False
        self.current_step = 0

        # Sequencer thread
        self.thread: Optional[threading.Thread] = None
        self.running = False

        # Flag to reset timing when starting playback
        self.reset_timing = False

        # Callback for triggering sounds (set by main app)
        self.on_step: Optional[Callable[[int], None]] = None

    def start(self):
        """Start the sequencer thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._sequencer_loop, daemon=True)
        self.thread.start()
        print("Sequencer started")

    def stop(self):
        """Stop the sequencer thread"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.thread = None
        print("Sequencer stopped")

    def play(self):
        """Start pattern playback"""
        self.is_playing = True
        self.current_step = 0
        self.reset_timing = True  # Signal the sequencer loop to reset timing

    def pause(self):
        """Pause pattern playback"""
        self.is_playing = False

    def reset(self):
        """Reset to step 0"""
        self.current_step = 0

    def set_bpm(self, bpm: int):
        """Set the BPM"""
        self.bpm = max(60, min(300, bpm))

    def set_swing(self, swing: float):
        """Set the swing amount (0.0 to 1.0, 0.5 = straight)"""
        self.swing = max(0.0, min(1.0, swing))

    def _calculate_step_duration(self, step_number: int) -> float:
        """
        Calculate the duration of a step in seconds, accounting for swing

        Args:
            step_number: The step number (0-15)

        Returns:
            Duration in seconds
        """
        # Base 16th note duration at current BPM
        base_duration = 60.0 / self.bpm / 4.0  # Quarter note / 4 = 16th note

        # Swing at 0.5 = straight (all steps equal)
        # Swing > 0.5 = shuffle (even steps longer, odd shorter)
        # Swing < 0.5 = reverse shuffle (even steps shorter, odd longer)

        # Always return base_duration for straight timing
        # (Swing feature disabled for now to prevent note skipping)
        return base_duration

    def _sequencer_loop(self):
        """Main sequencer loop running in separate thread"""
        start_time = time.time()
        step_start_time = start_time

        while self.running:
            current_time = time.time()

            # Check if we need to reset timing (when starting playback)
            if self.reset_timing:
                step_start_time = current_time
                self.reset_timing = False

            if self.is_playing and current_time >= step_start_time:
                # Trigger the current step via callback
                if self.on_step:
                    self.on_step(self.current_step)

                # Calculate the exact time this step should have started
                # This prevents drift by using absolute timing
                step_duration = self._calculate_step_duration(self.current_step)
                step_start_time += step_duration

                # Advance to next step
                self.current_step = (self.current_step + 1) % 16

                # If we've fallen too far behind, resync
                if current_time - step_start_time > 0.1:  # More than 100ms behind
                    step_start_time = current_time

            # Small sleep to prevent CPU spinning
            time.sleep(0.0005)  # 0.5ms sleep for better timing precision
