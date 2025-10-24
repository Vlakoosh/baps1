"""
Input Handler for BAPS-1 Sampler
Handles physical GPIO matrix on hardware, with pygame keyboard fallback for testing.
"""

import time
from enum import Enum
from typing import Callable, Optional, Dict, List, Set

try:
    import pygame  # type: ignore
except Exception:  # pragma: no cover - pygame may not be available on hardware
    pygame = None

try:
    import RPi.GPIO as GPIO  # type: ignore
except ImportError:  # pragma: no cover - ignored on development machines
    GPIO = None


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
    """Handles physical GPIO matrix or keyboard events for testing."""

    def __init__(self, use_gpio: bool = True):

        if use_gpio and GPIO is None:
            raise RuntimeError("GPIO input requested but RPi.GPIO is not available")
        if not use_gpio and pygame is None:
            raise RuntimeError("pygame is not available for keyboard fallback")

        self.use_gpio = use_gpio

        # Reverse mapping for pad lookups
        self.pad_buttons: List[Button] = [
            Button.PAD_1, Button.PAD_2, Button.PAD_3, Button.PAD_4,
            Button.PAD_5, Button.PAD_6, Button.PAD_7, Button.PAD_8,
            Button.PAD_9, Button.PAD_10, Button.PAD_11, Button.PAD_12,
            Button.PAD_13, Button.PAD_14, Button.PAD_15, Button.PAD_16,
        ]

        # Track button press times for hold detection
        self.button_press_times: Dict[Button, float] = {}
        self.hold_threshold = 0.3  # seconds

        # Currently held buttons
        self.held_buttons: Set[Button] = set()
        self._hold_fired: Set[Button] = set()

        # Callbacks
        self.on_button_press: Optional[Callable[[Button], None]] = None
        self.on_button_release: Optional[Callable[[Button], None]] = None
        self.on_button_hold: Optional[Callable[[Button], None]] = None
        self.on_encoder_turn: Optional[Callable[[int, int], None]] = None  # (encoder_num, direction)

        if self.use_gpio:
            print("InputHandler: using GPIO matrix backend")
            self._setup_gpio_matrix()
        else:
            print("InputHandler: using keyboard fallback backend")
            self._setup_keyboard_fallback()

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
        if self.use_gpio:
            self._poll_gpio_matrix()
            return True

        if pygame is None:
            return True

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
                        self._hold_fired.discard(button)
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
                    self._hold_fired.discard(button)

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
        """Cleanup resources"""
        if self.use_gpio and GPIO:
            GPIO.cleanup()
        elif pygame:
            pygame.quit()

    # --- Internal helpers -------------------------------------------------

    def _setup_keyboard_fallback(self):
        """Initialise pygame keyboard backend."""
        assert pygame is not None
        pygame.init()
        pygame.key.set_repeat()

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

    def _setup_gpio_matrix(self):
        """Initialise GPIO matrix backend."""
        assert GPIO is not None

        # BCM pin numbers for sensing columns/rows
        # Hardware wiring: columns feed the switch, current flows through diode into the row.
        self.column_pins: List[int] = [12, 16, 20, 21, 25]  # drive lines
        self.row_pins: List[int] = [5, 6, 13, 19, 26]       # sense lines

        # Physical layout mapping (row-major)
        self.matrix_map: List[List[Button]] = [
            [Button.SOUND, Button.PATTERN, Button.BPM, Button.TRIM, Button.VOLUME],
            [Button.PAD_1, Button.PAD_2, Button.PAD_3, Button.PAD_4, Button.RECORD],
            [Button.PAD_5, Button.PAD_6, Button.PAD_7, Button.PAD_8, Button.FX],
            [Button.PAD_9, Button.PAD_10, Button.PAD_11, Button.PAD_12, Button.PLAY],
            [Button.PAD_13, Button.PAD_14, Button.PAD_15, Button.PAD_16, Button.WRITE],
        ]

        GPIO.setmode(GPIO.BCM)
        for pin in self.column_pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        for pin in self.row_pins:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        # Rotary encoder pins (BCM)
        self.encoder_pins: Dict[int, tuple[int, int]] = {
            1: (17, 27),
            2: (22, 23),
        }
        for a_pin, b_pin in self.encoder_pins.values():
            GPIO.setup(a_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self._encoder_last: Dict[int, int] = {
            enc: self._read_encoder_state(enc) for enc in self.encoder_pins
        }
        self._encoder_accum: Dict[int, int] = {enc: 0 for enc in self.encoder_pins}

        self._active_buttons: Set[Button] = set()
        self.scan_delay = 0.00005  # seconds between column drive/read
        print(
            f"Matrix columns -> {self.column_pins}, rows -> {self.row_pins}, "
            f"encoders -> {self.encoder_pins}"
        )

    def _poll_gpio_matrix(self):
        """Scan the keypad matrix and emit callbacks."""
        assert GPIO is not None
        pressed_now: Set[Button] = set()
        current_time = time.time()

        for col_idx, col_pin in enumerate(self.column_pins):
            GPIO.output(col_pin, GPIO.HIGH)
            if self.scan_delay:
                time.sleep(self.scan_delay)

            for row_idx, row_pin in enumerate(self.row_pins):
                if GPIO.input(row_pin):
                    button = self.matrix_map[row_idx][col_idx]
                    pressed_now.add(button)

            GPIO.output(col_pin, GPIO.LOW)

        new_presses = pressed_now - self._active_buttons
        released = self._active_buttons - pressed_now

        for button in new_presses:
            self.button_press_times[button] = current_time
            self._hold_fired.discard(button)
            if self.is_pad_button(button):
                print(f"Pad pressed: {button.name}")
            if self.on_button_press:
                self.on_button_press(button)

        for button in released:
            self.button_press_times.pop(button, None)
            self._hold_fired.discard(button)
            if self.on_button_release:
                self.on_button_release(button)

        for button in pressed_now:
            press_time = self.button_press_times.get(button)
            if press_time is not None and (current_time - press_time) >= self.hold_threshold:
                if button not in self._hold_fired:
                    if self.on_button_hold:
                        self.on_button_hold(button)
                    self._hold_fired.add(button)

        self._active_buttons = pressed_now
        self.held_buttons = pressed_now.copy()
        self._poll_encoders()

    def _read_encoder_state(self, encoder_num: int) -> int:
        """Return current two-bit state for an encoder."""
        a_pin, b_pin = self.encoder_pins[encoder_num]
        a = GPIO.input(a_pin)
        b = GPIO.input(b_pin)
        return ((a & 1) << 1) | (b & 1)

    def _poll_encoders(self):
        """Decode quadrature encoder transitions and emit turn callbacks."""
        if self.on_encoder_turn is None:
            return

        transition_dir = {
            (0, 1): 1, (1, 3): 1, (3, 2): 1, (2, 0): 1,
            (0, 2): -1, (2, 3): -1, (3, 1): -1, (1, 0): -1,
        }

        for enc in self.encoder_pins:
            last = self._encoder_last[enc]
            current = self._read_encoder_state(enc)
            if current == last:
                continue

            direction = transition_dir.get((last, current))
            self._encoder_last[enc] = current

            if direction is None:
                continue

            self._encoder_accum[enc] += direction

            if self._encoder_accum[enc] >= 2:
                self.on_encoder_turn(enc, 1)
                print(f"Encoder {enc} turn: +1")
                self._encoder_accum[enc] -= 2
            elif self._encoder_accum[enc] <= -2:
                self.on_encoder_turn(enc, -1)
                print(f"Encoder {enc} turn: -1")
                self._encoder_accum[enc] += 2
