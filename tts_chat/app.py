#!/usr/bin/env python3
"""GUI chat client that streams LM Studio responses and speaks them in real time."""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog

try:
    from jinja2 import Environment
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit('jinja2 is required: pip install jinja2') from exc

JINJA_ENV = Environment(autoescape=False, extensions=['jinja2.ext.loopcontrols'])

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None

try:
    import simpleaudio as sa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sa = None

from kokoro import KPipeline

# Suppress noisy warnings originating from underlying model internals.
warnings.filterwarnings('ignore', message='dropout option adds dropout.*', category=UserWarning)
warnings.filterwarnings('ignore', message='`torch.nn.utils.weight_norm`.*', category=FutureWarning)

# Instantiate the TTS pipeline once so that subsequent calls are fast.
TTS_PIPELINE = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
SAMPLE_RATE = 24_000

THEME = {
    'bg': '#0e1621',
    'user_bg': '#1f5b9c',
    'user_fg': '#ffffff',
    'assistant_bg': '#1f2c3b',
    'assistant_fg': '#e4ecfa',
    'system_fg': '#7f9ab5',
    'status_fg': '#d1d8e0',
    'input_bg': '#17212b',
    'input_fg': '#e4ecfa',
    'button_bg': '#f1f5fa',
    'button_fg': '#11161f',
    'button_active_bg': '#d8e4f2',
    'button_active_fg': '#11161f',
    'panel_bg': '#151f2a',
    'panel_heading_fg': '#e4ecfa',
    'panel_label_fg': '#b3c1d1',
    'entry_bg': '#0f1a24',
    'entry_fg': '#e4ecfa',
}

PRESET_PATH = os.path.expanduser('~/.cache/lm-studio-tts/config-presets/llama-3-psychologist.preset.json')
PRESET_DIR = os.path.dirname(PRESET_PATH)
PRESET_EXTENSIONS = ('.preset.json', '.json')
DEFAULT_TEMPLATE = (
    "{{- bos_token if bos_token is defined else '' -}}\n"
    "{%- for message in messages %}\n"
    "  {{ '<|start_header_id|>' + (message.name or message.role) + '<|end_header_id|>\\n\\n' }}\n"
    "  {{ message.content | trim + '<|eot_id|>' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "  {{ '<|start_header_id|>' + assistant_api_name + '<|end_header_id|>\\n\\n' }}\n"
    "{%- endif %}"
)


class ChatClientError(Exception):
    """Raised when the chat client encounters a recoverable error."""


def play_audio_chunk(audio: np.ndarray) -> None:
    """Play a single chunk of audio using the best available backend."""
    if audio.size == 0:
        return

    if sd is not None:
        sd.play(audio, SAMPLE_RATE)
        sd.wait()
        return

    if sa is not None:
        audio_clipped = np.clip(audio, -1.0, 1.0)
        channels = 1 if audio_clipped.ndim == 1 else audio_clipped.shape[1]
        int_data = np.int16(audio_clipped * 32767)
        play_obj = sa.play_buffer(int_data, channels, 2, SAMPLE_RATE)
        play_obj.wait_done()
        return

    _play_via_tempfile(audio)


def _play_via_tempfile(audio: np.ndarray) -> None:
    """Fallback audio playback that writes to disk and delegates to the OS."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        file_path = tmp_file.name
        sf.write(file_path, audio, SAMPLE_RATE)

    try:
        if sys.platform == 'darwin':
            cmd = ['afplay', file_path]
        elif sys.platform.startswith('linux'):
            cmd = ['aplay', '-q', file_path]
        elif sys.platform.startswith('win'):
            cmd = [
                'powershell',
                '-NoProfile',
                '-Command',
                f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"
            ]
        else:
            raise ChatClientError('Unsupported platform for audio playback fallback.')

        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


def speak_text(text: str, voice: str, speed: float) -> None:
    """Convert text to speech and play it sequentially chunk by chunk."""
    if not text.strip():
        return

    generator = TTS_PIPELINE(text, voice=voice, speed=speed, split_pattern=r'\n+')
    for _, _, audio in generator:
        play_audio_chunk(np.asarray(audio, dtype=np.float32))


def stream_local_llm(
    prompt: str,
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    stop_event: Optional[threading.Event] = None,
    stop_sequences: Optional[List[str]] = None,
) -> Iterator[str]:
    """Yield assistant content chunks from the LM Studio completions endpoint."""
    url = base_url.rstrip('/') + '/v1/completions'
    payload: Dict[str, Any] = {
        'model': model,
        'prompt': prompt,
        'temperature': temperature,
        'stream': True,
    }
    if max_tokens > 0:
        payload['max_tokens'] = max_tokens
    if stop_sequences:
        payload['stop'] = stop_sequences

    response = requests.post(url, json=payload, timeout=timeout, stream=True)
    response.raise_for_status()
    response.encoding = 'utf-8'

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if stop_event and stop_event.is_set():
                break
            if not raw_line:
                continue
            if not raw_line.startswith('data: '):
                continue

            payload_line = raw_line[len('data: '):].strip()
            if payload_line == '[DONE]':
                break

            try:
                chunk = json.loads(payload_line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
                raise ChatClientError(f'Failed to parse stream chunk: {payload_line!r}') from exc

            choice = chunk.get('choices', [{}])[0]
            text = choice.get('text')
            if text:
                yield text
                continue

            delta = choice.get('delta', {})
            text = delta.get('text') or delta.get('content')
            if text:
                yield text
    finally:
        response.close()


def extract_speakable_segments(text: str, start_index: int) -> List[tuple[str, int]]:
    """Split completed sentences/newlines from text[start_index:] for playback."""
    segments: List[tuple[str, int]] = []
    idx = start_index
    length = len(text)

    while idx < length:
        char = text[idx]
        if char in '.!?':
            end = idx + 1
            while end < length and (text[end].isspace() or text[end] in '.:;!?)*"'):
                end += 1
            segment = text[start_index:end].strip()
            if segment:
                segments.append((segment, end))
            start_index = end
            idx = end
            continue

        if char == '\n':
            segment = text[start_index:idx].strip()
            if segment:
                segments.append((segment, idx + 1))
            start_index = idx + 1
            idx = start_index
            continue

        idx += 1

    return segments


class ChatApp:
    """Tkinter-based chat window that streams responses and speaks them."""

    @staticmethod
    def _decode_template_string(value: Optional[str]) -> str:
        if not value:
            return ''
        try:
            return bytes(value, 'utf-8').decode('unicode_escape')
        except Exception:
            return value

    @classmethod
    def _load_preset(cls, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as preset_file:
                data = json.load(preset_file)
        except FileNotFoundError:
            return {
                'path': path,
                'name': 'Default Preset',
                'user_role': 'patient',
                'assistant_role': 'psychologist',
                'system_prompt': '',
                'jinja_template': DEFAULT_TEMPLATE,
                'stop_sequences': [],
            }

        decode = cls._decode_template_string
        template = decode(data.get('jinja_template', DEFAULT_TEMPLATE)) or DEFAULT_TEMPLATE

        raw_stop = data.get('stop_sequences')
        if raw_stop is None:
            raw_stop = data.get('antiprompt', [])
        stop_sequences = [decode(item) for item in (raw_stop or [])]
        stop_sequences = [item for item in stop_sequences if item]

        return {
            'path': path,
            'name': data.get('name') or os.path.basename(path),
            'user_role': decode(data.get('user_role', 'user')) or 'user',
            'assistant_role': decode(data.get('assistant_role', 'assistant')) or 'assistant',
            'system_prompt': decode(data.get('system_prompt', '')),
            'jinja_template': template,
            'stop_sequences': stop_sequences,
            'bos_token': decode(data.get('bos_token', '')),
        }

    @staticmethod
    def _list_presets() -> List[str]:
        try:
            os.makedirs(PRESET_DIR, exist_ok=True)
            names = []
            for entry in os.listdir(PRESET_DIR):
                if any(entry.endswith(ext) for ext in PRESET_EXTENSIONS):
                    names.append(entry)
            names.sort(key=str.lower)
            return names
        except FileNotFoundError:
            return [os.path.basename(PRESET_PATH)]

    def _preset_display_from_filename(self, filename: str) -> str:
        if not self.preset_display_map:
            return filename
        for display, mapped in self.preset_display_map.items():
            if mapped == filename:
                return display
        return filename

    def _refresh_preset_menu(
        self,
        preselect_filename: Optional[str] = None,
        preferred_display: Optional[str] = None,
    ) -> None:
        preset_files = self._list_presets()
        new_map: Dict[str, str] = {}
        display_options: List[str] = []
        preferred_display_name = preferred_display

        if not preset_files:
            preset_files = [os.path.basename(PRESET_PATH)]

        for filename in preset_files:
            full_path = os.path.join(PRESET_DIR, filename)
            display_name = filename
            try:
                preset_data = self._load_preset(full_path)
                display_name = preset_data.get('name') or filename
            except Exception:
                display_name = filename

            unique_display = display_name
            counter = 2
            while unique_display in new_map:
                unique_display = f"{display_name} ({counter})"
                counter += 1

            new_map[unique_display] = filename
            display_options.append(unique_display)

            if filename == preselect_filename:
                preferred_display_name = unique_display

        self.preset_display_map = new_map

        if not display_options:
            display_options = ['']

        current_display = preferred_display_name or self._preset_display_from_filename(preselect_filename or os.path.basename(self.current_preset_path))
        if current_display not in display_options:
            current_display = display_options[0]

        self.preset_var.set(current_display)

        menu = self.preset_menu['menu']
        menu.delete(0, 'end')
        for display in display_options:
            menu.add_command(label=display, command=lambda value=display: self._on_preset_selected(value))

    @staticmethod
    def _sanitize_alias(alias: str, fallback: str) -> str:
        cleaned = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in alias.strip())
        if not cleaned:
            cleaned = fallback
        return cleaned[:64]

    def __init__(self, args: argparse.Namespace) -> None:
        os.makedirs(PRESET_DIR, exist_ok=True)
        self.args = args
        self.text_only = args.text_only
        self.current_preset = self._load_preset(PRESET_PATH)
        self.current_preset_path = PRESET_PATH
        self.jinja_template = self.current_preset.get('jinja_template', '')
        self.stop_sequences = self.current_preset.get('stop_sequences', [])
        self.bos_token = self.current_preset.get('bos_token', '')

        default_user_role = self.current_preset.get('user_role', 'user')
        default_assistant_role = self.current_preset.get('assistant_role', 'assistant')

        cli_user_role = args.user_role
        cli_assistant_role = args.assistant_role

        self.user_label_text = cli_user_role if cli_user_role else default_user_role
        self.assistant_label_text = cli_assistant_role if cli_assistant_role else default_assistant_role
        self.user_name_for_api = self._sanitize_alias(self.user_label_text, 'user')
        self.assistant_name_for_api = self._sanitize_alias(self.assistant_label_text, 'assistant')

        preset_default_system = self.current_preset.get('system_prompt', '')
        self.custom_system_prompt = (args.system.strip() if args.system is not None else preset_default_system)

        self.current_preset['user_role'] = self.user_label_text
        self.current_preset['assistant_role'] = self.assistant_label_text
        self.current_preset['system_prompt'] = self.custom_system_prompt
        self.current_preset['jinja_template'] = self.jinja_template
        self.current_preset['stop_sequences'] = self.stop_sequences
        self.current_preset['bos_token'] = self.bos_token
        self.messages: List[Dict[str, str]] = []
        self.system_message_entry: Optional[Dict[str, str]] = None

        self.root = tk.Tk()
        self.root.title('LM Studio Voice Chat')
        self.root.geometry('1200x720')
        self.root.minsize(900, 600)
        self.root.configure(bg=THEME['bg'])

        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.streaming_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.audio_disabled = args.text_only
        self.current_assistant_label: Optional[tk.Label] = None
        self.current_assistant_container: Optional[tk.Frame] = None
        self.current_assistant_bubble: Optional[tk.Frame] = None
        self.current_assistant_text: List[str] = []
        self.message_labels: List[tk.Label] = []
        self.message_label_map: Dict[int, tk.Label] = {}
        self.message_container_map: Dict[int, tk.Frame] = {}
        self.message_bubble_map: Dict[int, tk.Frame] = {}
        self.message_name_map: Dict[int, tk.Label] = {}
        self.preset_display_map: Dict[str, str] = {}
        self._settings_scroll_handlers: Dict[str, Any] = {}
        self.context_menu_target: Optional[Dict[str, str]] = None
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label='Edit message', command=self._start_edit_current_message)
        self.context_menu.add_command(label='Continue response', command=self._continue_current_message)
        self.context_menu.add_command(label='Delete message', command=self._delete_current_message)
        self.continuation_target: Optional[Dict[str, str]] = None
        self.continuation_notice: Optional[tk.Frame] = None
        self.edit_mode_entry: Optional[Dict[str, str]] = None
        self.wrap_length = 380
        self.chat_container: Optional[tk.Frame] = None
        self.current_assistant_name_label: Optional[tk.Label] = None
        self.process_queue_job: Optional[str] = None

        self._ensure_system_message(display_if_new=False)

        self.main_frame = tk.Frame(self.root, bg=THEME['bg'])
        self.main_frame.pack(fill='both', expand=True)

        self._build_ui(self.custom_system_prompt)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.process_queue_job = self.root.after(80, self.process_queue)
        self.root.bind('<Configure>', self._on_root_configure)

    def _build_ui(self, system_prompt: str) -> None:
        self.body_frame = tk.Frame(self.main_frame, bg=THEME['bg'])
        self.body_frame.pack(fill='both', expand=True, padx=10, pady=(10, 6))

        self.chat_container = tk.Frame(self.body_frame, bg=THEME['bg'])
        self.chat_container.pack(side='left', fill='both', expand=True)

        self.chat_canvas = tk.Canvas(self.chat_container, bg=THEME['bg'], highlightthickness=0, bd=0)
        self.chat_canvas.pack(side='left', fill='both', expand=True)
        self.chat_scrollbar = tk.Scrollbar(self.chat_container, orient='vertical', command=self.chat_canvas.yview)
        self.chat_scrollbar.pack(side='right', fill='y')
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.messages_frame = tk.Frame(self.chat_canvas, bg=THEME['bg'])
        self.chat_window = self.chat_canvas.create_window((0, 0), window=self.messages_frame, anchor='nw')
        self.messages_frame.bind('<Configure>', self._on_messages_configure)
        self._chat_scroll_handlers: Dict[str, Any] = {}
        self.chat_canvas.bind('<Enter>', lambda event: self._toggle_chat_scroll(True))
        self.chat_canvas.bind('<Leave>', lambda event: self._toggle_chat_scroll(False))

        self._build_settings_panel()

        input_frame = tk.Frame(self.main_frame, bg=THEME['bg'])
        input_frame.pack(side='bottom', fill='x', padx=10, pady=(0, 6))

        self.entry = tk.Text(
            input_frame,
            height=3,
            wrap='word',
            bg=THEME['input_bg'],
            fg=THEME['input_fg'],
            insertbackground=THEME['input_fg'],
            relief='flat',
            bd=0,
        )
        self.entry.pack(side='left', fill='both', expand=True)
        self.entry.configure(padx=10, pady=8, font=('Helvetica', 12))
        self.entry.bind('<Return>', self.on_enter_key)
        self.entry.bind('<Shift-Return>', lambda event: None)

        button_frame = tk.Frame(input_frame, bg=THEME['bg'])
        button_frame.pack(side='right', fill='y', padx=(6, 0))

        self.send_button = tk.Button(
            button_frame,
            text='Send',
            width=10,
            command=self.on_send,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            bd=0,
            padx=10,
            pady=6,
        )
        self.send_button.pack(fill='x')

        self.clear_button = tk.Button(
            button_frame,
            text='Clear',
            width=10,
            command=self.clear_entry,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            bd=0,
            padx=10,
            pady=6,
        )
        self.clear_button.pack(fill='x', pady=(6, 0))

        self.stop_button = tk.Button(
            button_frame,
            text='Stop',
            width=10,
            command=self.on_stop,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            bd=0,
            padx=10,
            pady=6,
            state=tk.DISABLED,
        )
        self.stop_button.pack(fill='x', pady=(6, 0))

        initial_status = self._ready_status()
        self.status_var = tk.StringVar(value=initial_status)
        status_label = tk.Label(
            self.main_frame,
            textvariable=self.status_var,
            anchor='w',
            bg=THEME['bg'],
            fg=THEME['status_fg'],
            padx=10,
        )
        status_label.pack(side='bottom', fill='x', padx=10, pady=(0, 10))

        if self.system_message_entry is not None:
            self.display_system_message(self.system_message_entry['content'], self.system_message_entry)

        self.entry.focus_set()

    def _build_settings_panel(self) -> None:
        self.settings_outer = tk.Frame(self.body_frame, bg=THEME['panel_bg'])
        self.settings_outer.pack(side='left', fill='y', padx=(12, 0), pady=(0, 12))

        self.settings_canvas = tk.Canvas(
            self.settings_outer,
            bg=THEME['panel_bg'],
            highlightthickness=0,
            bd=0,
            width=360,
        )
        self.settings_canvas.pack(side='left', fill='both', expand=False)

        self.settings_scrollbar = tk.Scrollbar(
            self.settings_outer,
            orient='vertical',
            command=self.settings_canvas.yview,
        )
        self.settings_scrollbar.pack(side='right', fill='y')
        self.settings_canvas.configure(yscrollcommand=self.settings_scrollbar.set)

        self.settings_frame = tk.Frame(self.settings_canvas, bg=THEME['panel_bg'], padx=12, pady=12)
        self.settings_window = self.settings_canvas.create_window((0, 0), window=self.settings_frame, anchor='nw')
        def _on_settings_configure(event: tk.Event) -> None:
            self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox('all'))
            canvas_width = self.settings_canvas.winfo_width()
            if canvas_width <= 1:
                canvas_width = event.width
            self.settings_canvas.itemconfig(self.settings_window, width=canvas_width)

        self.settings_frame.bind('<Configure>', _on_settings_configure)

        header = tk.Label(
            self.settings_frame,
            text='Settings',
            bg=THEME['panel_bg'],
            fg=THEME['panel_heading_fg'],
            font=('Helvetica', 12, 'bold'),
        )
        header.pack(anchor='w', pady=(0, 8))

        preset_label = tk.Label(
            self.settings_frame,
            text='Preset',
            bg=THEME['panel_bg'],
            fg=THEME['panel_label_fg'],
            font=('Helvetica', 10),
        )
        preset_label.pack(anchor='w')

        self.preset_var = tk.StringVar()
        self.preset_display_map: Dict[str, str] = {}
        self.preset_menu = tk.OptionMenu(
            self.settings_frame,
            self.preset_var,
            ''
        )
        self.preset_menu.configure(bg=THEME['entry_bg'], fg=THEME['entry_fg'], activebackground=THEME['button_active_bg'], activeforeground=THEME['button_active_fg'], relief='flat')
        self.preset_menu.pack(fill='x', pady=(2, 10))
        self._refresh_preset_menu(os.path.basename(self.current_preset_path))

        self.user_role_var = tk.StringVar(value=self.user_label_text)
        self.assistant_role_var = tk.StringVar(value=self.assistant_label_text)

        user_label = tk.Label(
            self.settings_frame,
            text='User role name',
            bg=THEME['panel_bg'],
            fg=THEME['panel_label_fg'],
            font=('Helvetica', 10),
        )
        user_label.pack(anchor='w', pady=(12, 2))

        self.user_role_entry = tk.Entry(
            self.settings_frame,
            textvariable=self.user_role_var,
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            insertbackground=THEME['entry_fg'],
            relief='flat',
        )
        self.user_role_entry.pack(fill='x')

        assistant_label = tk.Label(
            self.settings_frame,
            text='Assistant role name',
            bg=THEME['panel_bg'],
            fg=THEME['panel_label_fg'],
            font=('Helvetica', 10),
        )
        assistant_label.pack(anchor='w', pady=(12, 2))

        self.assistant_role_entry = tk.Entry(
            self.settings_frame,
            textvariable=self.assistant_role_var,
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            insertbackground=THEME['entry_fg'],
            relief='flat',
        )
        self.assistant_role_entry.pack(fill='x')

        system_label = tk.Label(
            self.settings_frame,
            text='System prompt',
            bg=THEME['panel_bg'],
            fg=THEME['panel_label_fg'],
            font=('Helvetica', 10),
        )
        system_label.pack(anchor='w', pady=(12, 2))

        self.system_prompt_text = tk.Text(
            self.settings_frame,
            height=6,
            wrap='word',
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            insertbackground=THEME['entry_fg'],
            relief='flat',
        )
        self.system_prompt_text.pack(fill='x')
        if self.custom_system_prompt:
            self.system_prompt_text.insert(tk.END, self.custom_system_prompt)

        template_label = tk.Label(
            self.settings_frame,
            text='Jinja template',
            bg=THEME['panel_bg'],
            fg=THEME['panel_label_fg'],
            font=('Helvetica', 10),
        )
        template_label.pack(anchor='w', pady=(12, 2))

        self.template_text = scrolledtext.ScrolledText(
            self.settings_frame,
            height=16,
            wrap='word',
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            insertbackground=THEME['entry_fg'],
            relief='flat',
        )
        self.template_text.pack(fill='both', expand=True)
        if self.jinja_template:
            self.template_text.insert(tk.END, self.jinja_template)

        button_row = tk.Frame(self.settings_frame, bg=THEME['panel_bg'])
        button_row.pack(fill='x', pady=(16, 0))

        apply_button = tk.Button(
            button_row,
            text='Apply Settings',
            command=self._apply_settings,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            pady=6,
        )
        apply_button.pack(side='left', fill='x', expand=True, padx=(0, 6))

        save_button = tk.Button(
            button_row,
            text='Save Preset',
            command=self._save_preset_dialog,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            pady=6,
        )
        save_button.pack(side='left', fill='x', expand=True)

        handlers = {
            '<MouseWheel>': self._on_settings_mousewheel,
            '<Button-4>': self._on_settings_scroll_up,
            '<Button-5>': self._on_settings_scroll_down,
        }
        self._settings_scroll_handlers = {}
        self.settings_canvas.bind('<Enter>', lambda event: self._toggle_settings_scroll(True, handlers))
        self.settings_canvas.bind('<Leave>', lambda event: self._toggle_settings_scroll(False, handlers))

    def set_input_state(self, enabled: bool) -> None:
        # Allow typing even while the model is responding; only gate the send button.
        self.entry.configure(state=tk.NORMAL)
        if self.edit_mode_entry is not None:
            # Editing controls the send button state separately
            self.send_button.configure(text='Edit', state=tk.NORMAL)
        else:
            self.send_button.configure(text='Send', state=tk.NORMAL if enabled else tk.DISABLED)
        if enabled:
            self.entry.focus_set()

    def clear_entry(self) -> None:
        if self.edit_mode_entry is not None:
            self._finish_edit_mode(cancel=True)
            return

        if self.entry['state'] == tk.NORMAL:
            self.entry.delete('1.0', tk.END)

    def _collect_pending_settings(self) -> Dict[str, str]:
        new_user = self.user_role_var.get().strip() or 'You'
        new_assistant = self.assistant_role_var.get().strip() or 'Assistant'
        new_system = self.system_prompt_text.get('1.0', tk.END).strip()
        new_template = self.template_text.get('1.0', tk.END)
        new_template = new_template.strip('\n') or DEFAULT_TEMPLATE
        return {
            'user_role': new_user,
            'assistant_role': new_assistant,
            'system_prompt': new_system,
            'jinja_template': new_template,
        }

    def _apply_settings(self) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            messagebox.showinfo('Settings', 'Please wait for the current response to finish before updating settings.')
            return

        pending = self._collect_pending_settings()

        new_user = pending['user_role']
        new_assistant = pending['assistant_role']
        new_system = pending['system_prompt']
        new_template = pending['jinja_template']

        self.custom_system_prompt = new_system
        self.jinja_template = new_template

        self.user_label_text = new_user
        self.assistant_label_text = new_assistant
        self.user_name_for_api = self._sanitize_alias(new_user, 'user')
        self.assistant_name_for_api = self._sanitize_alias(new_assistant, 'assistant')

        for entry in self.messages:
            role = entry.get('role')
            if role == 'user':
                entry['name'] = self.user_name_for_api
            elif role == 'assistant':
                entry['name'] = self.assistant_name_for_api

        self.custom_system_prompt = new_system
        self._update_message_headers()
        self._ensure_system_message(display_if_new=True)

        self.system_prompt_text.delete('1.0', tk.END)
        if self.custom_system_prompt:
            self.system_prompt_text.insert(tk.END, self.custom_system_prompt)
        self.template_text.delete('1.0', tk.END)
        self.template_text.insert(tk.END, self.jinja_template)

        if self.current_preset is not None:
            self.current_preset['user_role'] = self.user_label_text
            self.current_preset['assistant_role'] = self.assistant_label_text
            self.current_preset['system_prompt'] = self.custom_system_prompt
            self.current_preset['jinja_template'] = self.jinja_template

        self.user_role_var.set(self.user_label_text)
        self.assistant_role_var.set(self.assistant_label_text)

        self.status_var.set('Settings applied.')
        self.root.after(1500, self._update_status_if_idle)

    def _serialize_current_preset(self, name: str, pending: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if pending is None:
            user_role = self.user_label_text
            assistant_role = self.assistant_label_text
            system_prompt = self.custom_system_prompt
            template = self.jinja_template
        else:
            user_role = pending['user_role']
            assistant_role = pending['assistant_role']
            system_prompt = pending['system_prompt']
            template = pending['jinja_template']

        return {
            'name': name,
            'user_role': user_role,
            'assistant_role': assistant_role,
            'system_prompt': system_prompt,
            'jinja_template': template,
            'bos_token': self.bos_token,
            'stop_sequences': list(self.stop_sequences or []),
        }

    def _save_preset_dialog(self) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            messagebox.showinfo('Preset Busy', 'Please wait for the current response to finish before saving presets.')
            return

        current_name = None
        if self.current_preset is not None:
            current_name = self.current_preset.get('name')
        if not current_name:
            current_name = self._preset_display_from_filename(os.path.basename(self.current_preset_path))

        new_name = simpledialog.askstring('Save Preset', 'Preset name:', initialvalue=current_name, parent=self.root)
        if new_name is None:
            return

        new_name = new_name.strip()
        if not new_name:
            messagebox.showerror('Preset Error', 'Preset name cannot be empty.')
            return

        filename_base = re.sub(r'[^A-Za-z0-9._-]+', '_', new_name)
        if not filename_base:
            filename_base = 'preset'

        if not any(filename_base.endswith(ext) for ext in PRESET_EXTENSIONS):
            filename_base += PRESET_EXTENSIONS[0]

        os.makedirs(PRESET_DIR, exist_ok=True)
        path = os.path.join(PRESET_DIR, filename_base)
        if os.path.exists(path):
            overwrite = messagebox.askyesno('Overwrite Preset', f'Preset "{filename_base}" already exists. Overwrite?')
            if not overwrite:
                return

        pending = self._collect_pending_settings()

        data = self._serialize_current_preset(new_name, pending=pending)

        try:
            with open(path, 'w', encoding='utf-8') as preset_file:
                json.dump(data, preset_file, ensure_ascii=False, indent=2)
        except OSError as exc:
            messagebox.showerror('Preset Error', f'Failed to save preset:\n{exc}')
            return

        self.current_preset = data
        self.current_preset_path = path
        self.jinja_template = data['jinja_template']
        self.stop_sequences = data.get('stop_sequences', [])
        self.bos_token = data.get('bos_token', '')

        self._refresh_preset_menu(os.path.basename(path), preferred_display=new_name)

        self.status_var.set(f'Preset saved ({new_name})')
        self.root.after(1500, self._update_status_if_idle)

    def _on_preset_selected(self, preset_display: str) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            messagebox.showinfo('Preset Busy', 'Please wait for the current response to finish before switching presets.')
            self.preset_var.set(self._preset_display_from_filename(os.path.basename(self.current_preset_path)))
            return

        filename = self.preset_display_map.get(preset_display, preset_display)

        new_path = os.path.join(PRESET_DIR, filename)
        if not os.path.isfile(new_path):
            messagebox.showerror('Preset Error', f'Preset file not found:\n{new_path}')
            self.preset_var.set(self._preset_display_from_filename(os.path.basename(self.current_preset_path)))
            return

        try:
            new_preset = self._load_preset(new_path)
        except Exception as exc:
            messagebox.showerror('Preset Error', f'Failed to load preset:\n{exc}')
            self.preset_var.set(self._preset_display_from_filename(os.path.basename(self.current_preset_path)))
            return

        self.current_preset = new_preset
        self.current_preset_path = new_path

        self.jinja_template = new_preset.get('jinja_template', self.jinja_template)
        self.stop_sequences = new_preset.get('stop_sequences', [])
        self.bos_token = new_preset.get('bos_token', '')

        preset_default_system = new_preset.get('system_prompt', '')
        self.custom_system_prompt = preset_default_system
        self.system_prompt_text.delete('1.0', tk.END)
        if preset_default_system:
            self.system_prompt_text.insert(tk.END, preset_default_system)
        self.template_text.delete('1.0', tk.END)
        self.template_text.insert(tk.END, self.jinja_template)

        # Update roles from preset
        default_user_role = new_preset.get('user_role', self.user_label_text)
        default_assistant_role = new_preset.get('assistant_role', self.assistant_label_text)

        self.user_label_text = default_user_role
        self.assistant_label_text = default_assistant_role
        self.user_name_for_api = self._sanitize_alias(self.user_label_text, 'user')
        self.assistant_name_for_api = self._sanitize_alias(self.assistant_label_text, 'assistant')

        self.current_preset['user_role'] = self.user_label_text
        self.current_preset['assistant_role'] = self.assistant_label_text
        self.current_preset['system_prompt'] = self.custom_system_prompt
        self.current_preset['jinja_template'] = self.jinja_template
        self.current_preset['stop_sequences'] = self.stop_sequences

        self.user_role_var.set(self.user_label_text)
        self.assistant_role_var.set(self.assistant_label_text)

        # Refresh message metadata with new names
        for entry in self.messages:
            role = entry.get('role')
            if role == 'user':
                entry['name'] = self.user_name_for_api
            elif role == 'assistant':
                entry['name'] = self.assistant_name_for_api
            elif role == 'system' and self.system_message_entry is entry:
                entry['name'] = self._sanitize_alias('system', 'system')

        self._update_message_headers()
        self._ensure_system_message(display_if_new=True)

        self._refresh_preset_menu(os.path.basename(self.current_preset_path), preferred_display=new_preset.get('name'))

        display_name = new_preset.get('name') or filename
        self.status_var.set(f'Preset loaded ({display_name})')
        self.root.after(1500, self._update_status_if_idle)

    def _calc_wrap_length(self) -> int:
        width = self.chat_container.winfo_width() if self.chat_container else self.root.winfo_width()
        return max(260, width - 120)

    def _on_root_configure(self, _: tk.Event) -> None:
        new_wrap = self._calc_wrap_length()
        if new_wrap == self.wrap_length:
            return
        self.wrap_length = new_wrap
        for label in list(self.message_labels):
            if not int(label.winfo_exists()):
                self.message_labels.remove(label)
                continue
            label.configure(wraplength=self.wrap_length)
        container_width = (self.chat_container.winfo_width() if self.chat_container else self.root.winfo_width())
        self.chat_canvas.itemconfig(self.chat_window, width=max(100, container_width - 18))
        self._update_status_if_idle()
        self._scroll_to_bottom()

    def _on_mousewheel(self, event: tk.Event) -> None:
        if sys.platform == 'darwin':
            delta = event.delta
        else:
            delta = event.delta // 120
        self.chat_canvas.yview_scroll(int(-1 * delta), 'units')

    def _scroll_to_bottom(self) -> None:
        self.root.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))

    def _on_messages_configure(self, _: tk.Event) -> None:
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox('all'))
        container_width = (self.chat_container.winfo_width() if self.chat_container else self.root.winfo_width())
        self.chat_canvas.itemconfig(self.chat_window, width=max(100, container_width - 18))

    def _ready_status(self) -> str:
        if self.text_only:
            return 'Ready. (audio playback disabled)' if self.audio_disabled else 'Ready. (text only mode)'
        return 'Ready.'

    def _update_status_if_idle(self) -> None:
        if (
            self.streaming_thread
            and self.streaming_thread.is_alive()
            and self.edit_mode_entry is None
        ):
            return
        if self.edit_mode_entry is not None:
            return
        self.status_var.set(self._ready_status())

    def _update_message_headers(self) -> None:
        for entry in self.messages:
            key = id(entry)
            name_label = self.message_name_map.get(key)
            if not name_label or not int(name_label.winfo_exists()):
                continue
            role = entry.get('role')
            if role == 'user':
                name_label.configure(text=self.user_label_text)
            elif role == 'assistant':
                name_label.configure(text=self.assistant_label_text)
            elif role == 'system':
                name_label.configure(text='System')

    def _remove_message_ui(self, entry: Dict[str, Any]) -> None:
        key = id(entry)
        label = self.message_label_map.pop(key, None)
        container = self.message_container_map.pop(key, None)
        bubble = self.message_bubble_map.pop(key, None)
        name_label = self.message_name_map.pop(key, None)

        if label in self.message_labels:
            self.message_labels.remove(label)

        for widget in (name_label, label, bubble):
            if widget is not None and widget.winfo_exists():
                try:
                    widget.destroy()
                except Exception:
                    pass

        if container is not None and container.winfo_exists():
            try:
                container.destroy()
            except Exception:
                pass

    def _on_settings_mousewheel(self, event: tk.Event) -> str:
        if not hasattr(self, 'settings_canvas'):
            return 'break'
        if sys.platform == 'darwin':
            delta = -1 if event.delta > 0 else 1
        else:
            delta = -int(event.delta / 120) if event.delta else 0
        if delta:
            self.settings_canvas.yview_scroll(delta, 'units')
        return 'break'

    def _on_settings_scroll_up(self, _: tk.Event) -> str:
        if hasattr(self, 'settings_canvas'):
            self.settings_canvas.yview_scroll(-1, 'units')
        return 'break'

    def _on_settings_scroll_down(self, _: tk.Event) -> str:
        if hasattr(self, 'settings_canvas'):
            self.settings_canvas.yview_scroll(1, 'units')
        return 'break'

    def _toggle_chat_scroll(self, enable: bool) -> None:
        handlers = {
            '<MouseWheel>': self._on_mousewheel,
            '<Button-4>': lambda event: self.chat_canvas.yview_scroll(-1, 'units'),
            '<Button-5>': lambda event: self.chat_canvas.yview_scroll(1, 'units'),
        }
        if enable and not self._chat_scroll_handlers:
            for event, handler in handlers.items():
                self.chat_canvas.bind_all(event, handler)
            self._chat_scroll_handlers = handlers
        elif not enable and self._chat_scroll_handlers:
            for event in list(self._chat_scroll_handlers.keys()):
                self.chat_canvas.unbind_all(event)
            self._chat_scroll_handlers = {}

    def _toggle_settings_scroll(self, enable: bool, handlers: Dict[str, Any]) -> None:
        if enable and not self._settings_scroll_handlers:
            for event, handler in handlers.items():
                self.settings_canvas.bind_all(event, handler)
            self._settings_scroll_handlers = handlers
        elif not enable and self._settings_scroll_handlers:
            for event in list(self._settings_scroll_handlers.keys()):
                self.settings_canvas.unbind_all(event)
            self._settings_scroll_handlers = {}

    def _compose_system_prompt(self, custom_text: str) -> str:
        return custom_text.strip()

    def _ensure_system_message(self, *, display_if_new: bool) -> None:
        content = self._compose_system_prompt(self.custom_system_prompt)
        system_name = self._sanitize_alias('system', 'system')

        if self.current_preset is not None:
            self.current_preset['system_prompt'] = content

        if not content:
            if self.system_message_entry is not None:
                entry = self.system_message_entry
                self.system_message_entry = None
                try:
                    self.messages.remove(entry)
                except ValueError:
                    pass
                key = id(entry)
                label = self.message_label_map.pop(key, None)
                container = self.message_container_map.pop(key, None)
                self.message_bubble_map.pop(key, None)
                name_label = self.message_name_map.pop(key, None)
                if label in self.message_labels:
                    self.message_labels.remove(label)
                if container and int(container.winfo_exists()):
                    container.destroy()
                elif label is not None and int(label.winfo_exists()):
                    label.master.master.destroy()
                if name_label is not None and name_label.winfo_exists():
                    name_label.destroy()
            return

        if self.system_message_entry is None:
            entry = {'role': 'system', 'content': content, 'name': system_name}
            self.system_message_entry = entry
            self.messages.insert(0, entry)
            if display_if_new:
                self.display_system_message(content, entry)
        else:
            entry = self.system_message_entry
            entry['content'] = content
            entry['name'] = system_name
            key = id(entry)
            label = self.message_label_map.get(key)
            if label and int(label.winfo_exists()):
                label.configure(text=content)
            elif display_if_new:
                self.display_system_message(content, entry)

    def _build_prompt(self) -> str:
        template_source = self.jinja_template
        if not template_source:
            template_source = DEFAULT_TEMPLATE

        render_messages = [
            {
                'role': entry.get('role', ''),
                'name': entry.get('name', ''),
                'content': entry.get('content', ''),
            }
            for entry in self.messages
        ]

        try:
            template = JINJA_ENV.from_string(template_source)
        except Exception as exc:
            raise ChatClientError(f'Failed to compile preset template: {exc}') from exc

        context = {
            'messages': render_messages,
            'user_role': self.user_label_text,
            'assistant_role': self.assistant_label_text,
            'user_api_name': self.user_name_for_api,
            'assistant_api_name': self.assistant_name_for_api,
            'system_prompt': self.custom_system_prompt,
            'bos_token': self.bos_token,
            'add_generation_prompt': True,
        }

        try:
            return template.render(**context)
        except Exception as exc:
            raise ChatClientError(f'Failed to render preset template: {exc}') from exc

    @staticmethod
    def _leading_overlap(existing: str, new_text: str) -> int:
        max_overlap = min(len(existing), len(new_text))
        for size in range(max_overlap, 0, -1):
            if existing[-size:] == new_text[:size]:
                return size
        return 0

    def _create_message_bubble(
        self,
        role: str,
        text: str,
        *,
        italic: bool = False,
        message_entry: Optional[Dict[str, str]] = None,
    ) -> Tuple[tk.Frame, tk.Frame, tk.Label, tk.Label]:
        outer = tk.Frame(self.messages_frame, bg=THEME['bg'])
        outer.pack(fill='x', pady=4, padx=6)

        anchor = 'e' if role == 'user' else 'w'
        name_color = THEME['system_fg']
        if role == 'user':
            name_text = self.user_label_text
            bubble_bg = THEME['user_bg']
            text_color = THEME['user_fg']
        elif role == 'assistant':
            name_text = self.assistant_label_text
            bubble_bg = THEME['assistant_bg']
            text_color = THEME['assistant_fg']
        else:
            name_text = 'System'
            bubble_bg = '#152232'
            text_color = THEME['system_fg']

        container = tk.Frame(outer, bg=THEME['bg'])
        container.pack(anchor=anchor, fill='x')

        name_label = tk.Label(
            container,
            text=name_text,
            bg=THEME['bg'],
            fg=name_color,
            font=('Helvetica', 9, 'bold'),
        )
        name_label.pack(anchor=anchor, padx=8 if role != 'user' else 12, pady=(0, 2))

        bubble = tk.Frame(
            container,
            bg=bubble_bg,
            bd=0,
            relief='flat',
            highlightthickness=0,
        )
        bubble.pack(anchor=anchor, padx=8 if role != 'user' else 12)
        bubble.configure(padx=12, pady=8)

        font_style = ('Helvetica', 12, 'italic') if italic else ('Helvetica', 12)
        label = tk.Label(
            bubble,
            text=text,
            bg=bubble_bg,
            fg=text_color,
            font=font_style,
            justify='left',
            wraplength=self.wrap_length,
        )
        label.pack(anchor='w')

        self.message_labels.append(label)

        if message_entry is not None and message_entry.get('role') in {'user', 'assistant', 'system'}:
            self._register_message_widget(message_entry, label, bubble, outer, name_label)

        self._scroll_to_bottom()
        return outer, bubble, label, name_label

    def _register_message_widget(
        self,
        message_entry: Dict[str, str],
        label: tk.Label,
        bubble: Optional[tk.Frame],
        outer: Optional[tk.Frame],
        name_label: Optional[tk.Label],
    ) -> None:
        key = id(message_entry)
        self.message_label_map[key] = label
        if bubble is not None:
            self.message_bubble_map[key] = bubble
        if outer is not None:
            self.message_container_map[key] = outer
        if name_label is not None:
            self.message_name_map[key] = name_label

        def bind_widget(widget: Optional[tk.Widget]) -> None:
            if widget is None:
                return
            widget.bind('<Button-3>', lambda event, entry=message_entry: self.on_message_right_click(event, entry))
            widget.bind('<Button-2>', lambda event, entry=message_entry: self.on_message_right_click(event, entry))

        bind_widget(label)
        bind_widget(bubble)
        bind_widget(outer)
        bind_widget(name_label)

    def _enqueue_user_message(self, message_entry: Dict[str, str], *, display: bool) -> None:
        self.messages.append(message_entry)
        if display:
            self.display_user_message(message_entry)
        self.set_input_state(False)
        self.status_var.set('Awaiting response...')
        self.stop_event.clear()
        self.streaming_thread = threading.Thread(target=self.stream_response, daemon=True)
        self.streaming_thread.start()

    def display_system_message(self, text: str, message_entry: Optional[Dict[str, str]] = None) -> tk.Frame:
        container, bubble, label, name_label = self._create_message_bubble('system', text, italic=True, message_entry=message_entry)
        if message_entry is not None:
            key = id(message_entry)
            self.message_container_map[key] = container
            self.message_bubble_map[key] = bubble
            self.message_label_map[key] = label
            self.message_name_map[key] = name_label
        self._scroll_to_bottom()
        return container

    def display_user_message(self, message_entry: Dict[str, str]) -> None:
        self._create_message_bubble('user', message_entry['content'], message_entry=message_entry)

    def start_assistant_message(self, message_entry: Optional[Dict[str, str]] = None) -> None:
        if message_entry is not None:
            key = id(message_entry)
            label = self.message_label_map.get(key)
            container = self.message_container_map.get(key)
            bubble = self.message_bubble_map.get(key)
            name_label = self.message_name_map.get(key)
            if label is not None and container is not None:
                self.current_assistant_container = container
                self.current_assistant_bubble = bubble
                self.current_assistant_label = label
                self.current_assistant_name_label = name_label
                self.current_assistant_text = [message_entry.get('content', '')]
                return

        container, bubble, label, name_label = self._create_message_bubble('assistant', '')
        self.current_assistant_container = container
        self.current_assistant_bubble = bubble
        self.current_assistant_label = label
        self.current_assistant_name_label = name_label
        self.current_assistant_text = []

    def append_assistant_chunk(self, chunk: str) -> None:
        if not self.current_assistant_label:
            self.start_assistant_message()
        self.current_assistant_text.append(chunk)
        text = ''.join(self.current_assistant_text)
        if self.current_assistant_label:
            self.current_assistant_label.configure(text=text)
        self._scroll_to_bottom()

    def finalize_assistant_message(self, payload: Tuple[str, Optional[Dict[str, str]]] | str) -> None:
        if isinstance(payload, tuple):
            text, message_entry = payload
        else:
            text, message_entry = payload, None

        label = self.current_assistant_label
        container = self.current_assistant_container
        bubble = self.current_assistant_bubble

        if not label or not container:
            self.current_assistant_label = None
            self.current_assistant_container = None
            self.current_assistant_bubble = None
            self.current_assistant_text = []
            return

        if not text.strip():
            if label in self.message_labels:
                self.message_labels.remove(label)
            container.destroy()
            if message_entry is not None:
                key = id(message_entry)
                self.message_label_map.pop(key, None)
                self.message_container_map.pop(key, None)
                self.message_bubble_map.pop(key, None)
                self.message_name_map.pop(key, None)
        else:
            label.configure(text=text)
            if message_entry is not None:
                key = id(message_entry)
                if key not in self.message_label_map:
                    self._register_message_widget(
                        message_entry,
                        label,
                        bubble,
                        container,
                        self.current_assistant_name_label,
                    )

        self.current_assistant_label = None
        self.current_assistant_container = None
        self.current_assistant_bubble = None
        self.current_assistant_text = []
        self.current_assistant_name_label = None
        self._scroll_to_bottom()

    def on_enter_key(self, event) -> Optional[str]:
        if event.state & 0x1:  # allow Shift+Enter for newlines
            return None
        self.on_send()
        return 'break'

    def on_send(self) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            return

        content = self.entry.get('1.0', tk.END).strip()
        if not content:
            if self.edit_mode_entry is not None:
                messagebox.showwarning('Edit Message', 'Message cannot be empty.')
                return
            return

        if self.edit_mode_entry is not None:
            self._apply_edit(content)
            return

        self.entry.delete('1.0', tk.END)
        user_entry = {'role': 'user', 'content': content, 'name': self.user_name_for_api}
        self._enqueue_user_message(user_entry, display=True)

    def stream_response(self) -> None:
        continuation_entry = self.continuation_target
        self.continuation_target = None

        prefix = continuation_entry['content'] if continuation_entry else ''
        assistant_chunks: List[str] = [prefix] if prefix else []
        spoken_upto = len(prefix)
        error_status: Optional[str] = None
        assistant_entry: Optional[Dict[str, str]] = continuation_entry

        self._ensure_system_message(display_if_new=False)
        prompt_text = self._build_prompt()
        stop_sequences = self.stop_sequences

        self.queue.put(('assistant_begin', continuation_entry))
        self.queue.put(('streaming_state', True))

        try:
            for chunk in stream_local_llm(
                prompt_text,
                base_url=self.args.base_url,
                model=self.args.model,
                temperature=self.args.temperature,
                max_tokens=self.args.max_tokens,
                timeout=self.args.timeout,
                stop_event=self.stop_event,
                stop_sequences=stop_sequences,
            ):
                if not chunk:
                    continue

                existing_text = ''.join(assistant_chunks)
                overlap = self._leading_overlap(existing_text, chunk)
                trimmed_chunk = chunk[overlap:]
                if not trimmed_chunk:
                    continue

                assistant_chunks.append(trimmed_chunk)
                self.queue.put(('assistant_chunk', trimmed_chunk))

                if self.text_only:
                    continue

                cumulative = ''.join(assistant_chunks)
                for segment, new_index in extract_speakable_segments(cumulative, spoken_upto):
                    try:
                        speak_text(segment, voice=self.args.voice, speed=self.args.speed)
                    except Exception as exc:  # pragma: no cover - defensive path
                        self.queue.put(('system_message', f'[Audio Error] {exc}'))
                        self.queue.put(('status', 'Audio playback disabled after error.'))
                        self.queue.put(('text_only', True))
                        self.text_only = True
                        self.audio_disabled = True
                        break
                    spoken_upto = new_index
                if self.stop_event.is_set():
                    break
        except requests.RequestException as exc:
            if continuation_entry is None and self.messages:
                self.messages.pop()
            self.queue.put(('system_message', f'[Request Error] {exc}'))
            error_status = 'Request failed.'
        except ChatClientError as exc:
            if continuation_entry is None and self.messages:
                self.messages.pop()
            self.queue.put(('system_message', f'[Client Error] {exc}'))
            error_status = 'Client error.'
        else:
            assistant_reply = ''.join(assistant_chunks)
            if continuation_entry is not None:
                if not assistant_reply.strip():
                    assistant_reply = prefix
                continuation_entry['content'] = assistant_reply
                assistant_entry = continuation_entry
                if not self.messages or self.messages[-1] is not continuation_entry:
                    self.messages.append(continuation_entry)
            else:
                if not assistant_reply.strip():
                    if self.messages:
                        self.messages.pop()
                else:
                    assistant_entry = {
                        'role': 'assistant',
                        'content': assistant_reply,
                        'name': self.assistant_name_for_api,
                    }
                    self.messages.append(assistant_entry)
            if assistant_reply and not self.text_only and spoken_upto < len(assistant_reply):
                remaining = assistant_reply[spoken_upto:]
                if remaining.strip():
                    try:
                        speak_text(remaining, voice=self.args.voice, speed=self.args.speed)
                    except Exception as exc:  # pragma: no cover - defensive path
                        self.queue.put(('system_message', f'[Audio Error] {exc}'))
                        self.queue.put(('status', 'Audio playback disabled after error.'))
                        self.queue.put(('text_only', True))
                        self.text_only = True
                        self.audio_disabled = True
        finally:
            if self.stop_event.is_set() and error_status is None:
                self.queue.put(('status', 'Response interrupted.'))
            if error_status is not None:
                self.queue.put(('status', error_status))
            self.queue.put(('assistant_finalize', (''.join(assistant_chunks), assistant_entry)))
            self.queue.put(('status', self._ready_status()))
            self.queue.put(('clear_continue_notice', None))
            self.queue.put(('enable_input', None))
            self.queue.put(('streaming_state', False))
            self.stop_event.clear()
            self.streaming_thread = None

    def process_queue(self) -> None:
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break

            if kind == 'status':
                self.status_var.set(payload)
            elif kind == 'enable_input':
                self.set_input_state(True)
            elif kind == 'text_only':
                self.text_only = bool(payload)
                if self.text_only:
                    self.audio_disabled = True
            elif kind == 'assistant_begin':
                self.start_assistant_message(payload)
            elif kind == 'assistant_chunk':
                self.append_assistant_chunk(payload)
            elif kind == 'assistant_finalize':
                self.finalize_assistant_message(payload)
            elif kind == 'system_message':
                self.display_system_message(payload)
            elif kind == 'streaming_state':
                if payload:
                    self.stop_button.configure(state=tk.NORMAL)
                else:
                    self.stop_button.configure(state=tk.DISABLED)
            elif kind == 'clear_continue_notice':
                if self.continuation_notice and int(self.continuation_notice.winfo_exists()):
                    self.continuation_notice.destroy()
                self.continuation_notice = None

        self.process_queue_job = self.root.after(80, self.process_queue)

    def on_message_right_click(self, event: tk.Event, message_entry: Dict[str, str]) -> None:
        if self.edit_mode_entry is not None:
            return
        if self.streaming_thread and self.streaming_thread.is_alive():
            return
        if message_entry is None:
            return
        if message_entry.get('role') not in {'user', 'assistant', 'system'}:
            return

        can_continue = (
            message_entry.get('role') == 'assistant'
            and self.messages
            and self.messages[-1] is message_entry
        )
        self.context_menu.entryconfig('Continue response', state=tk.NORMAL if can_continue else tk.DISABLED)
        self.context_menu.entryconfig('Delete message', state=tk.NORMAL)

        self.context_menu_target = message_entry
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _start_edit_current_message(self) -> None:
        message_entry = self.context_menu_target
        if not message_entry:
            return

        # Ensure the message is still part of the active conversation
        if not any(entry is message_entry for entry in self.messages):
            messagebox.showinfo('Edit Message', 'This message is no longer part of the active conversation.')
            self.context_menu_target = None
            return

        current_text = message_entry.get('content', '')
        self.edit_mode_entry = message_entry
        self.entry.configure(state=tk.NORMAL)
        self.entry.delete('1.0', tk.END)
        self.entry.insert(tk.END, current_text)
        self.entry.focus_set()
        self.send_button.configure(text='Edit', state=tk.NORMAL)
        self.clear_button.configure(state=tk.NORMAL)
        self.status_var.set('Editing message... Press Enter to apply or Clear to cancel.')
        self.context_menu_target = None

    def _continue_current_message(self) -> None:
        message_entry = self.context_menu_target
        self.context_menu_target = None

        if not message_entry:
            return

        if self.edit_mode_entry is not None:
            return

        if self.streaming_thread and self.streaming_thread.is_alive():
            return

        if message_entry.get('role') != 'assistant':
            messagebox.showinfo('Continue Response', 'You can only continue assistant messages.')
            return

        if 'name' not in message_entry:
            message_entry['name'] = self.assistant_name_for_api

        if not self.messages or self.messages[-1] is not message_entry:
            messagebox.showinfo('Continue Response', 'Only the latest assistant reply can be continued.')
            return

        self.entry.configure(state=tk.NORMAL)
        self.entry.delete('1.0', tk.END)

        self.continuation_target = message_entry
        self.continuation_notice = self.display_system_message('Continuing response...')
        self.set_input_state(False)
        self.status_var.set('Continuing response...')
        self.stop_event.clear()

        self.streaming_thread = threading.Thread(target=self.stream_response, daemon=True)
        self.streaming_thread.start()

    def _delete_current_message(self) -> None:
        entry = self.context_menu_target
        self.context_menu_target = None

        if entry is None:
            return

        if self.edit_mode_entry is entry:
            self._finish_edit_mode(cancel=True)

        if self.continuation_target is entry:
            self.continuation_target = None

        if self.system_message_entry is entry:
            self.system_message_entry = None
            self.custom_system_prompt = ''
            self.system_prompt_text.delete('1.0', tk.END)
            self._ensure_system_message(display_if_new=False)

        removed_from_messages = False
        for idx, existing in enumerate(self.messages):
            if existing is entry:
                self.messages.pop(idx)
                removed_from_messages = True
                break

        self._remove_message_ui(entry)

        if removed_from_messages:
            self.status_var.set('Message deleted.')
            self.root.after(1500, self._update_status_if_idle)

    def _apply_edit(self, new_text: str) -> None:
        entry = self.edit_mode_entry
        if entry is None:
            return

        entry['content'] = new_text
        label = self.message_label_map.get(id(entry))
        if label is not None:
            label.configure(text=new_text)
            self._scroll_to_bottom()

        self._finish_edit_mode(cancel=False, status='Message updated.')

    def _finish_edit_mode(self, *, cancel: bool, status: Optional[str] = None) -> None:
        if cancel:
            self.status_var.set('Edit canceled.')
            self.root.after(1500, self._update_status_if_idle)
        elif status is not None:
            self.status_var.set(status)
            self.root.after(1500, self._update_status_if_idle)
        else:
            self.status_var.set(self._ready_status())

        self.edit_mode_entry = None
        self.entry.delete('1.0', tk.END)
        self.entry.configure(state=tk.NORMAL)
        self.send_button.configure(text='Send', state=tk.NORMAL)
        self.entry.focus_set()

    def on_close(self) -> None:
        self._toggle_chat_scroll(False)
        if hasattr(self, 'settings_canvas') and self._settings_scroll_handlers:
            self._toggle_settings_scroll(False, self._settings_scroll_handlers)

        if self.streaming_thread and self.streaming_thread.is_alive():
            self.stop_event.set()
            self.queue.put(('status', 'Stopping response...'))
            self.queue.put(('streaming_state', False))
            self.streaming_thread.join(timeout=2.0)
            self.streaming_thread = None

        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass

        if self.process_queue_job is not None:
            try:
                self.root.after_cancel(self.process_queue_job)
            except Exception:
                pass
            self.process_queue_job = None
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

    def on_stop(self) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            if not self.stop_event.is_set():
                self.stop_event.set()
                self.queue.put(('status', 'Stopping response...'))
