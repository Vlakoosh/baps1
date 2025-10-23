"""
Input Handler for BAPS-1 Sampler
Handles keyboard input for testing (maps to physical buttons on hardware)
"""

import pygame
from enum import Enum
from typing import Callable, Optional, Dict
import time


class Button(Enum):
    """Button types"""
    # Pads (1-16)
    PAD_1 = "1"
    PAD_2 = "2"
    PAD_3 = "3"
    PAD_4 = "4"
    PAD_5 = "q"
    PAD_6 = "w"
    PAD_7 = "e"
    PAD_8 = "r"
    PAD_9 = "a"
    PAD_10 = "s"
    PAD_11 = "d"
    PAD_12 = "f"
    PAD_13 = "z"
    PAD_14 = "x"
    PAD_15 = "c"
    PAD_16 = "v"

    # Mode buttons
    SOUND = "f1"
    PATTERN = "f2"
    BPM = "f3"
    WRITE = "f4"
    RECORD = "f5"
    VOLUME = "f6"
    TRIM = "f7"
    FX = "f8"
    PLAY = "space"

    # Encoders
    KNOB1_LEFT = "left"
    KNOB1_RIGHT = "right"
    KNOB2_UP = "up"
    KNOB2_DOWN = "down"


class InputHandler:
    """Handles keyboard input and button events"""

    def __init__(self):
        # Initialize pygame for keyboard input
        pygame.init()

        # Disable key repeat to prevent interference with hold detection
        pygame.key.set_repeat()

        # Key mappings (pygame key -> Button)
        self.key_map = {
            pygame.K_1: Button.PAD_1,
            pygame.K_2: Button.PAD_2,
            pygame.K_3: Button.PAD_3,
            pygame.K_4: Button.PAD_4,
            pygame.K_q: Button.PAD_5,
            pygame.K_w: Button.PAD_6,
            pygame.K_e: Button.PAD_7,
            pygame.K_r: Button.PAD_8,
            pygame.K_a: Button.PAD_9,
            pygame.K_s: Button.PAD_10,
            pygame.K_d: Button.PAD_11,
            pygame.K_f: Button.PAD_12,
            pygame.K_z: Button.PAD_13,
            pygame.K_x: Button.PAD_14,
            pygame.K_c: Button.PAD_15,
            pygame.K_v: Button.PAD_16,
            pygame.K_F1: Button.SOUND,
            pygame.K_F2: Button.PATTERN,
            pygame.K_F3: Button.BPM,
            pygame.K_F4: Button.WRITE,
            pygame.K_F5: Button.RECORD,
            pygame.K_F6: Button.VOLUME,
            pygame.K_F7: Button.TRIM,
            pygame.K_F8: Button.FX,
            pygame.K_SPACE: Button.PLAY,
            pygame.K_LEFT: Button.KNOB1_LEFT,
            pygame.K_RIGHT: Button.KNOB1_RIGHT,
            pygame.K_UP: Button.KNOB2_UP,
            pygame.K_DOWN: Button.KNOB2_DOWN,
        }

        # Reverse mapping for pad lookups
        self.pad_buttons = [
            Button.PAD_1, Button.PAD_2, Button.PAD_3, Button.PAD_4,
            Button.PAD_5, Button.PAD_6, Button.PAD_7, Button.PAD_8,
            Button.PAD_9, Button.PAD_10, Button.PAD_11, Button.PAD_12,
            Button.PAD_13, Button.PAD_14, Button.PAD_15, Button.PAD_16,
        ]

        # Track button press times for hold detection
        self.button_press_times: Dict[Button, float] = {}
        self.hold_threshold = 0.3  # seconds

        # Currently held buttons
        self.held_buttons = set()

        # Callbacks
        self.on_button_press: Optional[Callable[[Button], None]] = None
        self.on_button_release: Optional[Callable[[Button], None]] = None
        self.on_button_hold: Optional[Callable[[Button], None]] = None
        self.on_encoder_turn: Optional[Callable[[int, int], None]] = None  # (encoder_num, direction)

    def get_pad_number(self, button: Button) -> Optional[int]:
        """Get pad number (1-16) from button, or None if not a pad"""
        try:
            return self.pad_buttons.index(button) + 1
        except ValueError:
            return None

    def is_pad_button(self, button: Button) -> bool:
        """Check if button is a pad"""
        return button in self.pad_buttons

    def is_mode_button(self, button: Button) -> bool:
        """Check if button is a mode button"""
        mode_buttons = [
            Button.SOUND, Button.PATTERN, Button.BPM, Button.WRITE,
            Button.RECORD, Button.VOLUME, Button.TRIM, Button.FX, Button.PLAY
        ]
        return button in mode_buttons

    def is_encoder_button(self, button: Button) -> bool:
        """Check if button is an encoder turn"""
        encoder_buttons = [
            Button.KNOB1_LEFT, Button.KNOB1_RIGHT,
            Button.KNOB2_UP, Button.KNOB2_DOWN
        ]
        return button in encoder_buttons

    def poll_events(self):
        """Poll for input events and trigger callbacks"""
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to quit

            elif event.type == pygame.KEYDOWN:
                if event.key in self.key_map:
                    button = self.key_map[event.key]

                    # Handle encoder turns immediately
                    if self.is_encoder_button(button):
                        self._handle_encoder(button)
                    else:
                        # Track press time for hold detection
                        self.button_press_times[button] = current_time
                        self.held_buttons.add(button)

                        # Trigger press callback
                        if self.on_button_press:
                            self.on_button_press(button)

            elif event.type == pygame.KEYUP:
                if event.key in self.key_map:
                    button = self.key_map[event.key]

                    # Skip encoders (they don't have release events)
                    if self.is_encoder_button(button):
                        continue

                    # Remove from held buttons
                    self.held_buttons.discard(button)
                    self.button_press_times.pop(button, None)

                    # Trigger release callback
                    if self.on_button_release:
                        self.on_button_release(button)

        # Check for held buttons
        for button in list(self.held_buttons):
            if button in self.button_press_times:
                press_time = self.button_press_times[button]
                if current_time - press_time >= self.hold_threshold:
                    # Trigger hold callback once
                    if self.on_button_hold:
                        self.on_button_hold(button)
                    # Remove so we don't trigger again
                    self.button_press_times.pop(button, None)

        return True  # Continue running

    def _handle_encoder(self, button: Button):
        """Handle encoder turn"""
        if not self.on_encoder_turn:
            return

        if button == Button.KNOB1_LEFT:
            self.on_encoder_turn(1, -1)
        elif button == Button.KNOB1_RIGHT:
            self.on_encoder_turn(1, 1)
        elif button == Button.KNOB2_UP:
            self.on_encoder_turn(2, 1)
        elif button == Button.KNOB2_DOWN:
            self.on_encoder_turn(2, -1)

    def is_button_held(self, button: Button) -> bool:
        """Check if a button is currently being held"""
        return button in self.held_buttons

    def cleanup(self):
        """Cleanup pygame resources"""
        pygame.quit()
