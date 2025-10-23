"""
Display Manager for BAPS-1 Sampler
Handles screen rendering using luma.emulator (for testing) or luma.oled (for hardware)
"""

from luma.core.render import canvas
from luma.emulator.device import pygame
from luma.core.interface.serial import i2c
from PIL import ImageFont, ImageDraw
from typing import Optional, Sequence
import os
import numpy as np
import math


class DisplayManager:
    """Manages the OLED display for the BAPS-1 sampler"""

    def __init__(self, use_emulator: bool = True):
        """
        Initialize the display

        Args:
            use_emulator: If True, use pygame emulator. If False, use real hardware.
        """
        self.use_emulator = use_emulator
        self.width = 128
        self.height = 64

        # Initialize device
        if use_emulator:
            # Use pygame emulator for testing
            self.device = pygame(width=self.width, height=self.height, mode='1', scale=2)
            print("Display: Using emulator")
        else:
            # Use real I2C OLED (for Raspberry Pi)
            serial = i2c(port=1, address=0x3C)
            from luma.oled.device import ssd1309
            self.device = ssd1309(serial)
            print("Display: Using hardware SSD1309")

        # Try to load a font (default to built-in if not found)
        self.font_small = self._load_font(size=8)
        self.font_tiny = self._load_font(size=5)
        self.font_micro = self._load_font(size=4)
        self.font_medium = self._load_font(size=10)
        self.font_large = self._load_font(size=12)

        # Display state cache
        self.last_mode = None
        self.last_pad = None
        self.last_info = None

    def _load_font(self, size: int = 10):
        """Load a font, falling back to default if necessary"""
        try:
            # Try to load a TTF font
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except:
            pass

        # Fall back to default bitmap font
        return ImageFont.load_default()

    def clear(self):
        """Clear the display"""
        with canvas(self.device) as draw:
            draw.rectangle((0, 0, self.width, self.height), fill="black")

    def render(self, mode: str, pad_number: int, info_text: str = "",
               bar_value: float = 0.0, center_text: str = "", show_trim: bool = False,
               trim_start: float = 0.0, trim_end: float = 1.0, is_playing: bool = False):
        """
        Render the complete display

        Args:
            mode: Current mode (e.g., "FREE", "SOUND", "BPM")
            pad_number: Last pressed pad number (1-16)
            info_text: Text to show in top right (e.g., BPM, volume)
            bar_value: Value for the bottom bar (0.0 to 1.0)
            center_text: Text to show in the center area
            show_trim: If True, show trim region instead of level bar
            trim_start: Start position for trim (0.0 to 1.0)
            trim_end: End position for trim (0.0 to 1.0)
            is_playing: If True, show playback indicator
        """
        with canvas(self.device) as draw:
            # Top left: Mode and pad number (with play indicator)
            play_indicator = ">" if is_playing else ""
            mode_text = f"{play_indicator}{mode} P{pad_number:02d}"
            draw.text((2, 2), mode_text, fill="white", font=self.font_medium)

            # Top right: Info text
            if info_text:
                # Right-align the text
                bbox = draw.textbbox((0, 0), info_text, font=self.font_small)
                text_width = bbox[2] - bbox[0]
                draw.text((self.width - text_width - 2, 2), info_text, fill="white", font=self.font_small)

            # Center: Center text (file names, etc.)
            if center_text:
                # Center the text horizontally and vertically
                bbox = draw.textbbox((0, 0), center_text, font=self.font_medium)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (self.width - text_width) // 2
                y = (self.height - text_height) // 2
                draw.text((x, y), center_text, fill="white", font=self.font_medium)

            # Bottom: Bar
            bar_height = 8
            bar_y = self.height - bar_height - 2
            bar_x = 2
            bar_width = self.width - 4

            if show_trim:
                # Draw trim region
                # Background
                draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
                             outline="white", fill="black")

                # Trim region (highlighted area)
                trim_x_start = bar_x + int(trim_start * bar_width)
                trim_x_end = bar_x + int(trim_end * bar_width)
                draw.rectangle((trim_x_start, bar_y, trim_x_end, bar_y + bar_height),
                             fill="white")
            else:
                # Draw level bar
                # Background
                draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
                             outline="white", fill="black")

                # Level fill
                fill_width = int(bar_value * bar_width)
                if fill_width > 0:
                    draw.rectangle((bar_x, bar_y, bar_x + fill_width, bar_y + bar_height),
                                 fill="white")

    def render_sound_mode(self, pad_number: int, filename: str, file_index: int, file_list: list, choke_group: int = 0):
        """Render SOUND mode screen with vertical carousel and choke group"""
        with canvas(self.device) as draw:
            # Top left: Mode and pad number
            mode_text = f"SOUND P{pad_number:02d}"
            draw.text((2, 2), mode_text, fill="white", font=self.font_medium)

            # Top right: File counter
            info = f"{file_index + 1}/{len(file_list)}" if file_list else "0/0"
            bbox = draw.textbbox((0, 0), info, font=self.font_small)
            text_width = bbox[2] - bbox[0]
            draw.text((self.width - text_width - 2, 2), info, fill="white", font=self.font_small)

            # Display choke group at bottom right
            choke_text = f"CHK:{choke_group}"
            bbox = draw.textbbox((0, 0), choke_text, font=self.font_small)
            choke_width = bbox[2] - bbox[0]
            draw.text((self.width - choke_width - 2, self.height - 12), choke_text, fill="white", font=self.font_small)

            # Center carousel area
            carousel_y = 16
            carousel_height = 40

            # Show previous file (smaller, above)
            if file_index > 0 and file_list:
                prev_file = file_list[file_index - 1]
                # Truncate if too long
                if len(prev_file) > 16:
                    prev_file = prev_file[:13] + "..."
                prev_width = draw.textbbox((0, 0), prev_file, font=self.font_small)[2]
                prev_x = (self.width - prev_width) // 2
                draw.text((prev_x, carousel_y), prev_file, fill="white", font=self.font_small)

            # Show current file (larger, centered, with box)
            current_file = filename if filename else "[EMPTY]"
            # Truncate if too long
            if len(current_file) > 14:
                current_file = current_file[:11] + "..."

            current_y = carousel_y + 15
            bbox = draw.textbbox((0, 0), current_file, font=self.font_medium)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw box around current selection
            box_padding = 4
            box_x1 = (self.width - text_width) // 2 - box_padding
            box_y1 = current_y - 2
            box_x2 = box_x1 + text_width + box_padding * 2
            box_y2 = box_y1 + text_height + 4
            draw.rectangle((box_x1, box_y1, box_x2, box_y2), outline="white", fill="black")

            # Draw current file text
            current_x = (self.width - text_width) // 2
            draw.text((current_x, current_y), current_file, fill="white", font=self.font_medium)

            # Show next file (smaller, below)
            if file_index < len(file_list) - 1 and file_list:
                next_file = file_list[file_index + 1]
                # Truncate if too long
                if len(next_file) > 16:
                    next_file = next_file[:13] + "..."
                next_width = draw.textbbox((0, 0), next_file, font=self.font_small)[2]
                next_x = (self.width - next_width) // 2
                next_y = current_y + text_height + 4
                draw.text((next_x, next_y), next_file, fill="white", font=self.font_small)

    def _draw_character(self, draw, x: int, y: int, bobbing: bool = False):
        """Draw a simple pixel character that bobs to the beat"""
        # Character design (8x10 pixels)
        if bobbing:
            # Bobbing frame - head up
            # Head (circle) - moved up 1 pixel
            draw.ellipse((x+2, y-1, x+6, y+3), fill="white", outline="white")
            # Body (same position)
            draw.rectangle((x+3, y+4, x+5, y+8), fill="white")
            # Neck connecting head to body
            draw.line([(x+4, y+3), (x+4, y+4)], fill="white")
            # Arms
            draw.line([(x+1, y+5), (x+3, y+5)], fill="white")
            draw.line([(x+5, y+5), (x+7, y+5)], fill="white")
            # Legs
            draw.line([(x+3, y+8), (x+2, y+10)], fill="white")
            draw.line([(x+5, y+8), (x+6, y+10)], fill="white")
        else:
            # Normal frame - head at normal position
            # Head (circle)
            draw.ellipse((x+2, y, x+6, y+4), fill="white", outline="white")
            # Body
            draw.rectangle((x+3, y+4, x+5, y+8), fill="white")
            # Arms
            draw.line([(x+1, y+5), (x+3, y+5)], fill="white")
            draw.line([(x+5, y+5), (x+7, y+5)], fill="white")
            # Legs
            draw.line([(x+3, y+8), (x+2, y+10)], fill="white")
            draw.line([(x+5, y+8), (x+6, y+10)], fill="white")

    def render_free_mode(self, pad_number: int, level: float = 0.0, is_playing: bool = False,
                         pitch_all: float = 0.0, pitch_pad: float = 0.0, is_kit: bool = False,
                         current_step: int = 0):
        """Render FREE mode screen with play/pause icon, pitch info, and dancing character"""
        with canvas(self.device) as draw:
            # Top left: Mode and pad number (without play indicator prefix)
            mode_text = f"FREE P{pad_number:02d}"
            draw.text((2, 2), mode_text, fill="white", font=self.font_medium)

            # Left side: Dancing character (only when playing)
            # DISABLED FOR NOW - uncomment to re-enable
            # if is_playing:
            #     # Bob on every beat (steps 0, 4, 8, 12 = beats 1, 2, 3, 4)
            #     is_bobbing = current_step in [0, 4, 8, 12]
            #     char_x = 8
            #     char_y = 24
            #     self._draw_character(draw, char_x, char_y, bobbing=is_bobbing)

            # Top right: Pitch info
            if is_kit:
                # Show ALL and PAD pitch for kits
                if pitch_all != 0.0:
                    all_text = f"ALL:{pitch_all:+.2f}".rstrip('0').rstrip('.')
                    bbox = draw.textbbox((0, 0), all_text, font=self.font_small)
                    text_width = bbox[2] - bbox[0]
                    draw.text((self.width - text_width - 2, 2), all_text, fill="white", font=self.font_small)

                if pitch_pad != 0.0:
                    pad_text = f"PAD:{pitch_pad:+.2f}".rstrip('0').rstrip('.')
                    bbox = draw.textbbox((0, 0), pad_text, font=self.font_small)
                    text_width = bbox[2] - bbox[0]
                    draw.text((self.width - text_width - 2, 12), pad_text, fill="white", font=self.font_small)
            else:
                # Show ALL pitch for oneshots
                if pitch_all != 0.0:
                    all_text = f"ALL:{pitch_all:+.2f}".rstrip('0').rstrip('.')
                    bbox = draw.textbbox((0, 0), all_text, font=self.font_small)
                    text_width = bbox[2] - bbox[0]
                    draw.text((self.width - text_width - 2, 2), all_text, fill="white", font=self.font_small)

            # Center: Play/Pause icon
            icon_size = 16
            icon_x = (self.width - icon_size) // 2
            icon_y = (self.height - icon_size) // 2 - 5

            if is_playing:
                # Draw play triangle
                triangle = [
                    (icon_x, icon_y),  # Top left
                    (icon_x, icon_y + icon_size),  # Bottom left
                    (icon_x + icon_size, icon_y + icon_size // 2)  # Right point
                ]
                draw.polygon(triangle, fill="white", outline="white")
            else:
                # Draw pause bars
                bar_width = 5
                bar_spacing = 4
                draw.rectangle((icon_x, icon_y, icon_x + bar_width, icon_y + icon_size), fill="white")
                draw.rectangle((icon_x + bar_width + bar_spacing, icon_y,
                              icon_x + bar_width * 2 + bar_spacing, icon_y + icon_size), fill="white")

            # Bottom: Level bar
            bar_height = 8
            bar_y = self.height - bar_height - 2
            bar_x = 2
            bar_width_total = self.width - 4

            # Background
            draw.rectangle((bar_x, bar_y, bar_x + bar_width_total, bar_y + bar_height),
                         outline="white", fill="black")

            # Level fill
            fill_width = int(level * bar_width_total)
            if fill_width > 0:
                draw.rectangle((bar_x, bar_y, bar_x + fill_width, bar_y + bar_height),
                             fill="white")

    def render_pattern_mode(self, pad_number: int, pattern_number: int, chain_length: int = 0):
        """Render PATTERN mode screen"""
        info = f"C:{chain_length}" if chain_length > 0 else ""
        center = f"PTN {pattern_number}"

        self.render(
            mode="PATTERN",
            pad_number=pad_number,
            info_text=info,
            center_text=center,
            bar_value=0.0
        )

    def render_bpm_mode(self, pad_number: int, bpm: int, swing: float):
        """Render BPM mode screen"""
        swing_pct = int(swing * 100)
        info = f"S:{swing_pct}%"
        center = f"{bpm} BPM"

        self.render(
            mode="BPM",
            pad_number=pad_number,
            info_text=info,
            center_text=center,
            bar_value=0.0
        )

    def render_write_mode(self, pad_number: int, pattern_number: int, current_step: int = 0, steps_active: list = None):
        """Render WRITE mode screen with 4x4 grid"""
        if steps_active is None:
            steps_active = [False] * 16

        with canvas(self.device) as draw:
            # Top left: Mode and pad number
            mode_text = f"WRITE P{pad_number:02d}"
            draw.text((2, 2), mode_text, fill="white", font=self.font_medium)

            # Top right: Pattern number
            info = f"PTN{pattern_number}"
            bbox = draw.textbbox((0, 0), info, font=self.font_small)
            text_width = bbox[2] - bbox[0]
            draw.text((self.width - text_width - 2, 2), info, fill="white", font=self.font_small)

            # Draw 4x4 grid in center
            grid_size = 4
            cell_size = 8
            grid_spacing = 2
            total_grid_size = (cell_size * grid_size) + (grid_spacing * (grid_size - 1))

            # Center the grid
            grid_x = (self.width - total_grid_size) // 2
            grid_y = 20

            for row in range(grid_size):
                for col in range(grid_size):
                    step_idx = row * grid_size + col

                    x = grid_x + col * (cell_size + grid_spacing)
                    y = grid_y + row * (cell_size + grid_spacing)

                    # Draw cell
                    if steps_active[step_idx]:
                        # Filled for active steps
                        draw.rectangle((x, y, x + cell_size, y + cell_size), fill="white", outline="white")
                    else:
                        # Just outline for inactive steps
                        draw.rectangle((x, y, x + cell_size, y + cell_size), outline="white", fill="black")

                    # Highlight current step during playback
                    if step_idx == current_step:
                        # Draw a thicker border or inverse the colors
                        draw.rectangle((x-1, y-1, x + cell_size + 1, y + cell_size + 1), outline="white")

    def render_volume_mode(self, pad_number: int, pad_volume: float, kit_volume: float):
        """Render VOLUME mode screen"""
        channel_vol_pct = int(pad_volume * 100)
        master_vol_pct = int(kit_volume * 100)
        info = f"M:{master_vol_pct}%"
        center = f"CH:{channel_vol_pct}%"

        self.render(
            mode="VOLUME",
            pad_number=pad_number,
            info_text=info,
            center_text=center,
            bar_value=pad_volume
        )

    def render_fx_mode(
        self,
        submode: str,
        pad_number: int,
        is_playing: bool,
        active_pads: Sequence[int],
        held_pads: Sequence[int],
        pending_pads: Sequence[int],
        oneshot_hint: str,
        permanent_pad: int,
        effect_name: str,
        knob_left_value: float,
        knob_right_value: float,
        knob_left_text: str,
        knob_right_text: str,
        knob_hint: str
    ):
        """Render FX mode for both one-shot and permanent workflows."""
        sub_label = "SHOT" if submode == "ONE_SHOT" else "PERM"

        with canvas(self.device) as draw:
            play_indicator = ">" if is_playing else ""
            mode_text = f"{play_indicator}FX {sub_label}"
            draw.text((2, 2), mode_text, fill="white", font=self.font_medium)

            pad_label = f"PAD {pad_number:02d}"
            bbox_pad = draw.textbbox((0, 0), pad_label, font=self.font_small)
            draw.text(
                (self.width - (bbox_pad[2] - bbox_pad[0]) - 2, 2),
                pad_label,
                fill="white",
                font=self.font_small
            )

            if submode == "ONE_SHOT":
                active_text = "Active: " + (" ".join(f"{p:02d}" for p in active_pads) if active_pads else "--")
                held_text = "Held:   " + (" ".join(f"{p:02d}" for p in held_pads) if held_pads else "--")
                pending_text = "Next:   " + (" ".join(f"{p:02d}" for p in pending_pads) if pending_pads else "--")

                draw.text((4, 16), active_text, fill="white", font=self.font_small)
                draw.text((4, 28), held_text, fill="white", font=self.font_small)
                draw.text((4, 40), pending_text, fill="white", font=self.font_small)

                footer = oneshot_hint if oneshot_hint else "Hold pad to arm FX"
                bbox = draw.textbbox((0, 0), footer, font=self.font_tiny)
                text_width = bbox[2] - bbox[0]
                draw.text((self.width - text_width - 2, self.height - 10),
                          footer, fill="white", font=self.font_tiny)

            else:  # PERMANENT
                # Effect name centered above knobs
                effect_label = effect_name.upper() if effect_name else "NO EFFECT"
                bbox_effect = draw.textbbox((0, 0), effect_label, font=self.font_medium)
                effect_width = bbox_effect[2] - bbox_effect[0]
                draw.text(((self.width - effect_width) // 2, 18), effect_label, fill="white", font=self.font_medium)

                if knob_hint:
                    bbox_hint = draw.textbbox((0, 0), knob_hint, font=self.font_tiny)
                    draw.text(((self.width - (bbox_hint[2] - bbox_hint[0])) // 2, 28),
                              knob_hint, fill="white", font=self.font_tiny)

                # Knobs for HP / LP controls
                knob_radius = 12
                left_center = (34, 38)
                right_center = (94, 38)

                self._draw_knob(draw, left_center, knob_radius, knob_left_value, knob_left_text)
                self._draw_knob(draw, right_center, knob_radius, knob_right_value, knob_right_text)

    def _draw_knob(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        value: float,
        value_text: str
    ):
        """Draw a knob with markers, indicator, and value."""
        cx, cy = center

        # Outer circle
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline="white",
            fill="black"
        )

        # Min/max markers positioned at bottom corners
        marker_angles = (135, 45)
        for angle in marker_angles:
            rad = math.radians(angle)
            inner_x = cx + math.cos(rad) * (radius - 1)
            inner_y = cy + math.sin(rad) * (radius - 1)
            outer_x = cx + math.cos(rad) * (radius + 2)
            outer_y = cy + math.sin(rad) * (radius + 2)
            draw.line([(inner_x, inner_y), (outer_x, outer_y)], fill="white")

        # Indicator line + dot
        sweep_angle = 135 + value * 270.0  # sweep across top arc
        sweep_rad = math.radians(sweep_angle)
        line_x = cx + math.cos(sweep_rad) * (radius - 3)
        line_y = cy + math.sin(sweep_rad) * (radius - 3)
        draw.line([(cx, cy), (line_x, line_y)], fill="white")
        draw.ellipse((line_x - 1, line_y - 1, line_x + 1, line_y + 1), fill="white")

        # Value below knob
        value_bbox = draw.textbbox((0, 0), value_text, font=self.font_micro)
        value_width = value_bbox[2] - value_bbox[0]
        value_height = value_bbox[3] - value_bbox[1]
        value_x = cx - value_width // 2
        value_y = cy + radius + 2
        max_y = self.height - value_height - 2
        if value_y > max_y:
            value_y = max_y
        draw.text((value_x, value_y), value_text, fill="white", font=self.font_micro)

    def render_trim_mode(self, pad_number: int, start_time: float, duration: float, total_duration: float, audio_data=None):
        """Render TRIM mode screen with waveform"""
        # Calculate positions
        if total_duration > 0:
            trim_start = start_time / total_duration
            trim_end = min(1.0, (start_time + duration) / total_duration)
        else:
            trim_start = 0.0
            trim_end = 0.0

        with canvas(self.device) as draw:
            # Top left: Mode and pad number
            mode_text = f"TRIM P{pad_number:02d}"
            draw.text((2, 2), mode_text, fill="white", font=self.font_medium)

            # Top right: Duration info
            info = f"{duration:.2f}s"
            bbox = draw.textbbox((0, 0), info, font=self.font_small)
            text_width = bbox[2] - bbox[0]
            draw.text((self.width - text_width - 2, 2), info, fill="white", font=self.font_small)

            # Middle: Waveform visualization
            if audio_data is not None and len(audio_data) > 0:
                waveform_y = 18
                waveform_height = 28
                waveform_x = 2
                waveform_width = self.width - 4

                # Downsample audio to fit display width
                samples_per_pixel = max(1, len(audio_data) // waveform_width)

                # Draw waveform
                for x in range(waveform_width):
                    # Get chunk of samples for this pixel
                    start_idx = x * samples_per_pixel
                    end_idx = min(start_idx + samples_per_pixel, len(audio_data))

                    if end_idx > start_idx:
                        # Get min and max of this chunk
                        chunk = audio_data[start_idx:end_idx]
                        min_val = float(chunk.min())
                        max_val = float(chunk.max())

                        # Map to display coordinates (center line + amplitude)
                        center_y = waveform_y + waveform_height // 2
                        y1 = int(center_y - (max_val * waveform_height // 2))
                        y2 = int(center_y - (min_val * waveform_height // 2))

                        # Draw vertical line for this pixel
                        draw.line([(waveform_x + x, y1), (waveform_x + x, y2)], fill="white")

                # Draw trim region markers
                trim_x_start = waveform_x + int(trim_start * waveform_width)
                trim_x_end = waveform_x + int(trim_end * waveform_width)

                # Draw vertical lines at trim boundaries
                draw.line([(trim_x_start, waveform_y), (trim_x_start, waveform_y + waveform_height)], fill="white", width=2)
                draw.line([(trim_x_end, waveform_y), (trim_x_end, waveform_y + waveform_height)], fill="white", width=2)
            else:
                # No waveform, show start time text
                center_text = f"S:{start_time:.2f}s"
                bbox = draw.textbbox((0, 0), center_text, font=self.font_medium)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (self.width - text_width) // 2
                y = 25
                draw.text((x, y), center_text, fill="white", font=self.font_medium)

            # Bottom: Trim region bar
            bar_height = 8
            bar_y = self.height - bar_height - 2
            bar_x = 2
            bar_width = self.width - 4

            # Background
            draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
                         outline="white", fill="black")

            # Trim region (highlighted area)
            trim_x_start = bar_x + int(trim_start * bar_width)
            trim_x_end = bar_x + int(trim_end * bar_width)
            draw.rectangle((trim_x_start, bar_y, trim_x_end, bar_y + bar_height),
                         fill="white")

    def render_record_mode(self, pad_number: int, is_recording: bool = False):
        """Render RECORD mode screen"""
        center = "RECORDING..." if is_recording else "HOLD PAD"

        self.render(
            mode="RECORD",
            pad_number=pad_number,
            info_text="",
            center_text=center,
            bar_value=0.0
        )

    def show_message(self, message: str, duration: float = 1.0):
        """Show a temporary message"""
        with canvas(self.device) as draw:
            # Center the message
            bbox = draw.textbbox((0, 0), message, font=self.font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2
            draw.text((x, y), message, fill="white", font=self.font_large)

    def cleanup(self):
        """Cleanup display resources"""
        try:
            self.clear()
            self.device.cleanup()
        except:
            pass
