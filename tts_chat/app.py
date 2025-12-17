from __future__ import annotations

import os
import queue
import re
import sys
import threading
import warnings
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import requests
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog

from .audio import AudioPlayer
from .completions import stream_completions
from .constants import (
    APP_NAME,
    DEFAULT_PRESET_PATH,
    DEFAULT_WINDOW_GEOMETRY,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    DEFAULT_TEMPLATE,
    PRESET_EXTENSIONS,
    PRESET_ROOT,
    THEME,
)
from .exceptions import ChatClientError
from .kokoro_local import list_local_voices, mix_voice_tensors, save_voice_tensor
from .preset import Preset, ensure_preset_dir, list_presets, load_preset, save_preset
from .prompt import build_jinja_environment, render_prompt
from .utils import extract_speakable_segments, leading_overlap

warnings.filterwarnings('ignore', message='dropout option adds dropout.*', category=UserWarning)
warnings.filterwarnings('ignore', message='`torch.nn.utils.weight_norm`.*', category=FutureWarning)


class ChatApp:
    """Tkinter application orchestrating chat, presets, and audio playback."""

    def __init__(self, args) -> None:
        ensure_preset_dir()

        self.args = args
        self.text_only = args.text_only
        self.base_url = args.base_url
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.timeout = args.timeout
        self.voice = args.voice
        self.speed = args.speed
        self.seed: Optional[int] = getattr(args, 'seed', None)

        self.audio_player: Optional[AudioPlayer] = None if args.text_only else AudioPlayer()
        self.jinja_env = build_jinja_environment()

        preset = load_preset(DEFAULT_PRESET_PATH)
        if args.system:
            preset = replace(preset, system_prompt=args.system.strip())
        if args.user_role:
            preset = replace(preset, user_role=args.user_role.strip() or preset.user_role)
        if args.assistant_role:
            preset = replace(preset, assistant_role=args.assistant_role.strip() or preset.assistant_role)

        self.preset: Preset = preset
        self.current_preset_path: str = preset.path or DEFAULT_PRESET_PATH

        self.messages: List[Dict[str, Any]] = []
        self.system_message_entry: Optional[Dict[str, Any]] = None
        self.continuation_target: Optional[Dict[str, Any]] = None
        self.continuation_notice: Optional[tk.Frame] = None
        self.edit_mode_entry: Optional[Dict[str, Any]] = None
        self.queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
        self.streaming_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.audio_disabled = args.text_only

        self.current_assistant_label: Optional[tk.Label] = None
        self.current_assistant_container: Optional[tk.Frame] = None
        self.current_assistant_bubble: Optional[tk.Frame] = None
        self.current_assistant_name_label: Optional[tk.Label] = None
        self.current_assistant_text: List[str] = []

        self.message_labels: List[tk.Label] = []
        self.message_label_map: Dict[int, tk.Label] = {}
        self.message_container_map: Dict[int, tk.Frame] = {}
        self.message_bubble_map: Dict[int, tk.Frame] = {}
        self.message_name_map: Dict[int, tk.Label] = {}
        self.preset_display_map: Dict[str, str] = {}
        self._settings_scroll_handlers: Dict[str, Any] = {}
        self._chat_scroll_handlers: Dict[str, Any] = {}
        self.process_queue_job: Optional[str] = None
        self.context_menu_target: Optional[Dict[str, Any]] = None
        self.context_menu: Optional[tk.Menu] = None
        self.wrap_length = 380
        self.mix_window: Optional[tk.Toplevel] = None
        self.mix_voice1_var: Optional[tk.StringVar] = None
        self.mix_voice2_var: Optional[tk.StringVar] = None
        self.mix_alpha_var: Optional[tk.DoubleVar] = None
        self.mix_test_text: Optional[tk.Text] = None
        self.mix_name_var: Optional[tk.StringVar] = None

        self._update_aliases()
        self._ensure_system_message(display_if_new=False)

        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.geometry(DEFAULT_WINDOW_GEOMETRY)
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.root.configure(bg=THEME['bg'])
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label='Edit message', command=self._start_edit_current_message)
        self.context_menu.add_command(label='Continue response', command=self._continue_current_message)
        self.context_menu.add_command(label='Delete message', command=self._delete_current_message)

        self.main_frame = tk.Frame(self.root, bg=THEME['bg'])
        self.main_frame.pack(fill='both', expand=True)

        self._build_ui()
        self.process_queue_job = self.root.after(80, self.process_queue)
        self.root.bind('<Configure>', self._on_root_configure)

    # ------------------------------------------------------------------
    # UI helpers

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.main_frame, bg=THEME['bg'])
        toolbar.pack(fill='x', padx=10, pady=(10, 0))

        mix_button = tk.Button(
            toolbar,
            text='Mix Voices',
            command=self._open_mix_window,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            bd=0,
            padx=10,
            pady=4,
        )
        mix_button.pack(side='left')

        self.body_frame = tk.Frame(self.main_frame, bg=THEME['bg'])
        self.body_frame.pack(fill='both', expand=True, padx=10, pady=(6, 6))

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

        self.chat_canvas.bind('<Enter>', lambda event: self._toggle_chat_scroll(True))
        self.chat_canvas.bind('<Leave>', lambda event: self._toggle_chat_scroll(False))

        self._build_settings_panel()

        input_frame = tk.Frame(self.main_frame, bg=THEME['bg'])
        input_frame.pack(fill='x', padx=10, pady=(0, 6))

        self.entry = tk.Text(
            input_frame,
            height=3,
            wrap='word',
            bg=THEME['input_bg'],
            fg=THEME['input_fg'],
            insertbackground=THEME['input_fg'],
            relief='flat',
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
        status_label.pack(fill='x', padx=10, pady=(0, 10))

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
        self.preset_menu = tk.OptionMenu(self.settings_frame, self.preset_var, '')
        self.preset_menu.configure(
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
        )
        self.preset_menu.pack(fill='x', pady=(2, 10))
        self._refresh_preset_menu(os.path.basename(self.current_preset_path))

        voice_label = tk.Label(
            self.settings_frame,
            text='Voice',
            bg=THEME['panel_bg'],
            fg=THEME['panel_label_fg'],
            font=('Helvetica', 10),
        )
        voice_label.pack(anchor='w', pady=(4, 2))

        self.voice_options = self._load_voice_options()
        initial_voice = self.voice if self.voice in self.voice_options else (self.voice_options[0] if self.voice_options else self.voice)
        self.voice_var = tk.StringVar(value=initial_voice)
        self.voice_menu = tk.OptionMenu(self.settings_frame, self.voice_var, *(self.voice_options or ['']))
        self.voice_menu.configure(
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
        )
        self.voice_menu.pack(fill='x', pady=(0, 8))

        self.user_role_var = tk.StringVar(value=self.preset.user_role)
        self.assistant_role_var = tk.StringVar(value=self.preset.assistant_role)

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
        if self.preset.system_prompt:
            self.system_prompt_text.insert(tk.END, self.preset.system_prompt)

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
        if self.preset.jinja_template:
            self.template_text.insert(tk.END, self.preset.jinja_template)

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
        self.settings_canvas.bind('<Enter>', lambda event: self._toggle_settings_scroll(True, handlers))
        self.settings_canvas.bind('<Leave>', lambda event: self._toggle_settings_scroll(False, handlers))

    # ------------------------------------------------------------------
    # General helpers

    def _load_voice_options(self) -> List[str]:
        try:
            voices = list_local_voices()
        except Exception:
            voices = []
        voices = sorted(set(voices))
        if self.voice and self.voice not in voices:
            voices.insert(0, self.voice)
        return voices

    def _refresh_voice_menu_options(self, selected: Optional[str] = None) -> None:
        options = self._load_voice_options()
        target = selected or self.voice
        if target not in options and options:
            target = options[0]
        self.voice_options = options
        self.voice_var.set(target)
        menu = self.voice_menu['menu']
        menu.delete(0, 'end')
        for opt in options or ['']:
            menu.add_command(label=opt, command=lambda v=opt: self.voice_var.set(v))

    @staticmethod
    def _sanitize_alias(alias: str, fallback: str) -> str:
        cleaned = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in alias.strip())
        if not cleaned:
            cleaned = fallback
        return cleaned[:64]

    def _update_aliases(self) -> None:
        self.user_alias = self._sanitize_alias(self.preset.user_role, 'user')
        self.assistant_alias = self._sanitize_alias(self.preset.assistant_role, 'assistant')

    def _ready_status(self) -> str:
        if self.text_only:
            return 'Ready. (audio playback disabled)' if self.audio_disabled else 'Ready. (text only mode)'
        return 'Ready.'

    def _update_status_if_idle(self) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            return
        if self.edit_mode_entry is not None:
            return
        self.status_var.set(self._ready_status())

    def _toggle_chat_scroll(self, enable: bool) -> None:
        handlers = {
            '<MouseWheel>': self._on_chat_mousewheel,
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

    def _open_mix_window(self) -> None:
        if self.mix_window is not None and self.mix_window.winfo_exists():
            self.mix_window.deiconify()
            self.mix_window.lift()
            self.mix_window.focus_force()
            return

        self.mix_window = tk.Toplevel(self.root)
        self.mix_window.title('Mix Voices')
        self.mix_window.geometry('360x260')
        self.mix_window.configure(bg=THEME['bg'])
        self.mix_window.transient(self.root)

        msg = tk.Label(
            self.mix_window,
            text='Voice mixer coming soon.',
            bg=THEME['bg'],
            fg=THEME['status_fg'],
            font=('Helvetica', 12),
            padx=20,
            pady=20,
        )
        msg.pack(expand=True)

        def _on_close() -> None:
            if self.mix_window is not None and self.mix_window.winfo_exists():
                self.mix_window.destroy()
            self.mix_window = None

        self.mix_window.protocol('WM_DELETE_WINDOW', _on_close)

        voices = self._load_voice_options()
        default_voice1 = voices[0] if voices else ''
        default_voice2 = voices[1] if len(voices) > 1 else (voices[0] if voices else '')

        selection_frame = tk.Frame(self.mix_window, bg=THEME['bg'])
        selection_frame.pack(fill='x', padx=14, pady=(12, 6))

        tk.Label(selection_frame, text='Voice A', bg=THEME['bg'], fg=THEME['panel_label_fg']).grid(row=0, column=0, sticky='w')
        tk.Label(selection_frame, text='Voice B', bg=THEME['bg'], fg=THEME['panel_label_fg']).grid(row=1, column=0, sticky='w')

        self.mix_voice1_var = tk.StringVar(value=default_voice1)
        self.mix_voice2_var = tk.StringVar(value=default_voice2)
        voice_menu_a = tk.OptionMenu(selection_frame, self.mix_voice1_var, *(voices or ['']))
        voice_menu_b = tk.OptionMenu(selection_frame, self.mix_voice2_var, *(voices or ['']))
        for menu in (voice_menu_a, voice_menu_b):
            menu.configure(
                bg=THEME['entry_bg'],
                fg=THEME['entry_fg'],
                activebackground=THEME['button_active_bg'],
                activeforeground=THEME['button_active_fg'],
                relief='flat',
                highlightthickness=0,
            )
        voice_menu_a.grid(row=0, column=1, sticky='ew', padx=(6, 0))
        voice_menu_b.grid(row=1, column=1, sticky='ew', padx=(6, 0))
        selection_frame.columnconfigure(1, weight=1)

        alpha_frame = tk.Frame(self.mix_window, bg=THEME['bg'])
        alpha_frame.pack(fill='x', padx=14, pady=(6, 4))
        tk.Label(alpha_frame, text='Blend (Voice A ↔ Voice B)', bg=THEME['bg'], fg=THEME['panel_label_fg']).pack(anchor='w')
        self.mix_alpha_var = tk.DoubleVar(value=0.5)
        alpha_scale = tk.Scale(
            alpha_frame,
            variable=self.mix_alpha_var,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient='horizontal',
            length=240,
            bg=THEME['bg'],
            fg=THEME['status_fg'],
            troughcolor=THEME['panel_bg'],
            highlightthickness=0,
        )
        alpha_scale.pack(fill='x')

        test_frame = tk.Frame(self.mix_window, bg=THEME['bg'])
        test_frame.pack(fill='both', expand=True, padx=14, pady=(6, 4))
        tk.Label(test_frame, text='Test text', bg=THEME['bg'], fg=THEME['panel_label_fg']).pack(anchor='w')
        self.mix_test_text = tk.Text(
            test_frame,
            height=3,
            wrap='word',
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            insertbackground=THEME['entry_fg'],
            relief='flat',
        )
        self.mix_test_text.pack(fill='x')
        self.mix_test_text.insert(tk.END, "This is a mixed voice preview.")

        play_btn = tk.Button(
            test_frame,
            text='Play Preview',
            command=self._play_mix_preview,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            padx=10,
            pady=6,
        )
        play_btn.pack(anchor='e', pady=(6, 0))

        name_frame = tk.Frame(self.mix_window, bg=THEME['bg'])
        name_frame.pack(fill='x', padx=14, pady=(6, 4))
        tk.Label(name_frame, text='New voice name', bg=THEME['bg'], fg=THEME['panel_label_fg']).pack(anchor='w')
        self.mix_name_var = tk.StringVar()
        tk.Entry(
            name_frame,
            textvariable=self.mix_name_var,
            bg=THEME['entry_bg'],
            fg=THEME['entry_fg'],
            insertbackground=THEME['entry_fg'],
            relief='flat',
        ).pack(fill='x')

        button_row = tk.Frame(self.mix_window, bg=THEME['bg'])
        button_row.pack(fill='x', padx=14, pady=(10, 12))
        save_btn = tk.Button(
            button_row,
            text='Save',
            command=self._save_mixed_voice,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            padx=10,
            pady=6,
        )
        cancel_btn = tk.Button(
            button_row,
            text='Cancel',
            command=_on_close,
            bg=THEME['button_bg'],
            fg=THEME['button_fg'],
            activebackground=THEME['button_active_bg'],
            activeforeground=THEME['button_active_fg'],
            relief='flat',
            padx=10,
            pady=6,
        )
        save_btn.pack(side='right', padx=(6, 0))
        cancel_btn.pack(side='right')

    def _on_chat_mousewheel(self, event: tk.Event) -> str:
        delta = event.delta
        if sys.platform != 'darwin':
            delta //= 120
        self.chat_canvas.yview_scroll(int(-delta), 'units')
        return 'break'

    def _on_settings_mousewheel(self, event: tk.Event) -> str:
        delta = event.delta
        if sys.platform != 'darwin':
            delta //= 120
        self.settings_canvas.yview_scroll(int(-delta), 'units')
        return 'break'

    def _on_settings_scroll_up(self, _: tk.Event) -> str:
        self.settings_canvas.yview_scroll(-1, 'units')
        return 'break'

    def _on_settings_scroll_down(self, _: tk.Event) -> str:
        self.settings_canvas.yview_scroll(1, 'units')
        return 'break'

    def _on_root_configure(self, _: tk.Event) -> None:
        self.wrap_length = max(260, self.chat_container.winfo_width() - 120)
        for label in list(self.message_labels):
            if label.winfo_exists():
                label.configure(wraplength=self.wrap_length)
        self.chat_canvas.itemconfig(self.chat_window, width=max(100, self.chat_container.winfo_width() - 18))
        self._scroll_to_bottom()

    def _on_messages_configure(self, _: tk.Event) -> None:
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox('all'))
        container_width = self.chat_container.winfo_width()
        self.chat_canvas.itemconfig(self.chat_window, width=max(100, container_width - 18))

    def _play_mix_preview(self) -> None:
        if self.audio_player is None or self.audio_disabled:
            messagebox.showinfo('Audio disabled', 'Enable audio to preview voices.')
            return
        if not self.mix_voice1_var or not self.mix_voice2_var or not self.mix_alpha_var:
            return
        v1 = self.mix_voice1_var.get().strip()
        v2 = self.mix_voice2_var.get().strip()
        if not v1 or not v2:
            messagebox.showerror('Missing voices', 'Select both voices to mix.')
            return
        alpha = self.mix_alpha_var.get()
        try:
            mix_tensor = mix_voice_tensors(v1, v2, alpha)
        except Exception as exc:
            messagebox.showerror('Mix error', f'Failed to mix voices:\n{exc}')
            return

        preview_name = '__preview_mix__'
        self.audio_player._pipeline.voices[preview_name] = mix_tensor  # type: ignore[attr-defined]
        text_widget = self.mix_test_text
        test_text = text_widget.get('1.0', tk.END).strip() if text_widget else ''
        if not test_text:
            test_text = "This is a mixed voice preview."
        try:
            self.audio_player.speak_text(test_text, voice=preview_name, speed=self.speed)
        except Exception as exc:
            messagebox.showerror('Playback error', f'Failed to play preview:\n{exc}')

    def _save_mixed_voice(self) -> None:
        if not (self.mix_voice1_var and self.mix_voice2_var and self.mix_alpha_var and self.mix_name_var):
            return
        name = self.mix_name_var.get().strip()
        if not name:
            messagebox.showerror('Missing name', 'Please provide a name for the new voice.')
            return
        v1 = self.mix_voice1_var.get().strip()
        v2 = self.mix_voice2_var.get().strip()
        if not v1 or not v2:
            messagebox.showerror('Missing voices', 'Select both voices to mix.')
            return

        try:
            mix_tensor = mix_voice_tensors(v1, v2, self.mix_alpha_var.get())
        except Exception as exc:
            messagebox.showerror('Mix error', f'Failed to mix voices:\n{exc}')
            return

        try:
            save_voice_tensor(name, mix_tensor)
        except Exception as exc:
            messagebox.showerror('Save error', f'Failed to save new voice:\n{exc}')
            return

        # Clear cached voice so it reloads from disk if present
        try:
            if self.audio_player is not None:
                self.audio_player._pipeline.voices.pop(name, None)  # type: ignore[attr-defined]
        except Exception:
            pass

        self._refresh_voice_menu_options(selected=name)
        # Update mix window options too
        voices = self._load_voice_options()
        for var in (self.mix_voice1_var, self.mix_voice2_var):
            if var.get() not in voices and voices:
                var.set(voices[0])

        messagebox.showinfo('Voice saved', f'New voice "{name}" saved.')

    # ------------------------------------------------------------------
    # Preset handling

    def _refresh_preset_menu(
        self,
        preselect_filename: Optional[str] = None,
        preferred_display: Optional[str] = None,
    ) -> None:
        preset_files = list_presets()
        new_map: Dict[str, str] = {}
        display_options: List[str] = []
        preferred_display_name = preferred_display

        if not preset_files:
            preset_files = [os.path.basename(DEFAULT_PRESET_PATH)]

        for filename in preset_files:
            full_path = os.path.join(PRESET_ROOT, filename)
            try:
                preset_obj = load_preset(full_path)
                display_name = preset_obj.name or filename
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

        current_display = preferred_display_name
        if not current_display and preselect_filename:
            current_display = self._preset_display_from_filename(preselect_filename)
        if not current_display and display_options:
            current_display = display_options[0]

        self.preset_var.set(current_display)

        menu = self.preset_menu['menu']
        menu.delete(0, 'end')
        for display in display_options:
            menu.add_command(label=display, command=lambda value=display: self._on_preset_selected(value))

    def _preset_display_from_filename(self, filename: str) -> str:
        for display, mapped in self.preset_display_map.items():
            if mapped == filename:
                return display
        return filename

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
        self.preset = replace(
            self.preset,
            user_role=pending['user_role'],
            assistant_role=pending['assistant_role'],
            system_prompt=pending['system_prompt'],
            jinja_template=pending['jinja_template'],
        )
        self._update_aliases()

        for entry in self.messages:
            role = entry.get('role')
            if role == 'user':
                entry['name'] = self.user_alias
            elif role == 'assistant':
                entry['name'] = self.assistant_alias

        self.voice = self.voice_var.get().strip() or self.voice

        self.user_role_var.set(self.preset.user_role)
        self.assistant_role_var.set(self.preset.assistant_role)
        self.system_prompt_text.delete('1.0', tk.END)
        if self.preset.system_prompt:
            self.system_prompt_text.insert(tk.END, self.preset.system_prompt)
        self.template_text.delete('1.0', tk.END)
        self.template_text.insert(tk.END, self.preset.jinja_template)

        self._update_message_headers()
        self._ensure_system_message(display_if_new=True)

        self.status_var.set('Settings applied.')
        self.root.after(1500, self._update_status_if_idle)

    def _save_preset_dialog(self) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            messagebox.showinfo('Preset Busy', 'Please wait for the current response to finish before saving presets.')
            return

        current_name = self.preset.name or os.path.basename(self.current_preset_path)
        new_name = simpledialog.askstring('Save Preset', 'Preset name:', initialvalue=current_name, parent=self.root)
        if new_name is None:
            return

        new_name = new_name.strip()
        if not new_name:
            messagebox.showerror('Preset Error', 'Preset name cannot be empty.')
            return

        pending = self._collect_pending_settings()
        preset_to_save = replace(
            self.preset,
            name=new_name,
            user_role=pending['user_role'],
            assistant_role=pending['assistant_role'],
            system_prompt=pending['system_prompt'],
            jinja_template=pending['jinja_template'],
        )

        filename_base = re.sub(r'[^A-Za-z0-9._-]+', '_', new_name) or 'preset'
        if not any(filename_base.endswith(ext) for ext in PRESET_EXTENSIONS):
            filename_base += PRESET_EXTENSIONS[0]

        target_path = os.path.join(PRESET_ROOT, filename_base)
        if os.path.exists(target_path):
            overwrite = messagebox.askyesno('Overwrite Preset', f'Preset "{filename_base}" already exists. Overwrite?')
            if not overwrite:
                return

        preset_to_save = replace(preset_to_save, path=target_path)
        save_preset(preset_to_save, path=target_path)

        self.preset = preset_to_save
        self.current_preset_path = target_path
        self._refresh_preset_menu(os.path.basename(target_path), preferred_display=new_name)

        self.status_var.set(f'Preset saved ({new_name})')
        self.root.after(1500, self._update_status_if_idle)

    def _on_preset_selected(self, preset_display: str) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            messagebox.showinfo('Preset Busy', 'Please wait for the current response to finish before switching presets.')
            self.preset_var.set(self._preset_display_from_filename(os.path.basename(self.current_preset_path)))
            return

        filename = self.preset_display_map.get(preset_display, preset_display)
        preset_path = os.path.join(PRESET_ROOT, filename)
        if not os.path.isfile(preset_path):
            messagebox.showerror('Preset Error', f'Preset file not found:\n{preset_path}')
            self.preset_var.set(self._preset_display_from_filename(os.path.basename(self.current_preset_path)))
            return

        try:
            new_preset = load_preset(preset_path)
        except Exception as exc:
            messagebox.showerror('Preset Error', f'Failed to load preset:\n{exc}')
            self.preset_var.set(self._preset_display_from_filename(os.path.basename(self.current_preset_path)))
            return

        self.preset = replace(self.preset, **new_preset.__dict__)
        self.current_preset_path = preset_path
        self._update_aliases()

        for entry in self.messages:
            role = entry.get('role')
            if role == 'user':
                entry['name'] = self.user_alias
            elif role == 'assistant':
                entry['name'] = self.assistant_alias

        self.user_role_var.set(self.preset.user_role)
        self.assistant_role_var.set(self.preset.assistant_role)
        self.system_prompt_text.delete('1.0', tk.END)
        if self.preset.system_prompt:
            self.system_prompt_text.insert(tk.END, self.preset.system_prompt)
        self.template_text.delete('1.0', tk.END)
        self.template_text.insert(tk.END, self.preset.jinja_template)

        self._update_message_headers()
        self._ensure_system_message(display_if_new=True)
        self._refresh_preset_menu(os.path.basename(preset_path), preferred_display=self.preset.name)

        self.status_var.set(f'Preset loaded ({self.preset.name})')
        self.root.after(1500, self._update_status_if_idle)

    # ------------------------------------------------------------------
    # Message utilities

    def _scroll_to_bottom(self) -> None:
        self.root.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))

    def _create_message_bubble(
        self,
        role: str,
        text: str,
        *,
        italic: bool = False,
        message_entry: Optional[Dict[str, Any]] = None,
    ) -> Tuple[tk.Frame, tk.Frame, tk.Label, tk.Label]:
        outer = tk.Frame(self.messages_frame, bg=THEME['bg'])
        outer.pack(fill='x', pady=4, padx=6)

        anchor = 'e' if role == 'user' else 'w'
        name_color = THEME['system_fg']
        if role == 'user':
            name_text = self.preset.user_role
            bubble_bg = THEME['user_bg']
            text_color = THEME['user_fg']
        elif role == 'assistant':
            name_text = self.preset.assistant_role
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
        message_entry: Dict[str, Any],
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

        for widget in (label, bubble, outer, name_label):
            if widget is None:
                continue
            widget.bind('<Button-3>', lambda event, entry=message_entry: self.on_message_right_click(event, entry))
            widget.bind('<Button-2>', lambda event, entry=message_entry: self.on_message_right_click(event, entry))

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

    def _update_message_headers(self) -> None:
        for entry in self.messages:
            key = id(entry)
            name_label = self.message_name_map.get(key)
            if not name_label or not name_label.winfo_exists():
                continue
            role = entry.get('role')
            if role == 'user':
                name_label.configure(text=self.preset.user_role)
            elif role == 'assistant':
                name_label.configure(text=self.preset.assistant_role)
            elif role == 'system':
                name_label.configure(text='System')

    def _ensure_system_message(self, *, display_if_new: bool) -> None:
        content = (self.preset.system_prompt or '').strip()
        if not content:
            if self.system_message_entry is not None:
                entry = self.system_message_entry
                self.system_message_entry = None
                try:
                    self.messages.remove(entry)
                except ValueError:
                    pass
                self._remove_message_ui(entry)
            return

        if self.system_message_entry is None:
            entry = {'role': 'system', 'name': 'system', 'content': content}
            self.system_message_entry = entry
            self.messages.insert(0, entry)
            if display_if_new:
                self.display_system_message(content, entry)
        else:
            self.system_message_entry['content'] = content
            label = self.message_label_map.get(id(self.system_message_entry))
            if label and label.winfo_exists():
                label.configure(text=content)
            elif display_if_new:
                self.display_system_message(content, self.system_message_entry)

    def _render_prompt(
        self,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        return render_prompt(
            self.jinja_env,
            self.preset,
            messages,
            user_api_name=self.user_alias,
            assistant_api_name=self.assistant_alias,
            add_generation_prompt=add_generation_prompt,
        )

    def _build_prompt(self) -> str:
        return self._render_prompt(self.messages, add_generation_prompt=True)

    @staticmethod
    def _leading_overlap(existing: str, new_text: str) -> int:
        return leading_overlap(existing, new_text)

    # ------------------------------------------------------------------
    # Event handlers (sending, editing, etc.)

    def set_input_state(self, enabled: bool) -> None:
        self.entry.configure(state=tk.NORMAL)
        if self.edit_mode_entry is not None:
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

    def on_enter_key(self, event) -> Optional[str]:
        if event.state & 0x1:
            return None
        self.on_send()
        return 'break'

    def on_send(self) -> None:
        if self.streaming_thread and self.streaming_thread.is_alive():
            return

        content = self.entry.get('1.0', tk.END).strip()
        if not content:
            if self.edit_mode_entry is not None:
                messagebox.showinfo('Edit Message', 'Message cannot be empty.')
            return

        if self.edit_mode_entry is not None:
            self._apply_edit(content)
            return

        self.entry.delete('1.0', tk.END)
        user_entry = {'role': 'user', 'name': self.user_alias, 'content': content}
        self.messages.append(user_entry)
        self.display_user_message(user_entry)
        self.set_input_state(False)
        self.status_var.set('Awaiting response...')
        self.stop_event.clear()
        self.streaming_thread = threading.Thread(target=self.stream_response, daemon=True)
        self.streaming_thread.start()

    def _collect_audio_player(self) -> Optional[AudioPlayer]:
        if self.audio_player is None and not self.text_only:
            self.audio_player = AudioPlayer()
        return self.audio_player

    def stream_response(self) -> None:
        continuation_entry = self.continuation_target
        self.continuation_target = None

        prefix = continuation_entry['content'] if continuation_entry else ''
        assistant_chunks: List[str] = [prefix] if prefix else []
        spoken_upto = len(prefix)
        error_status: Optional[str] = None
        assistant_entry: Optional[Dict[str, Any]] = continuation_entry

        stop_sequences = self.preset.stop_sequences
        if continuation_entry is not None:
            base_messages = [msg for msg in self.messages if msg is not continuation_entry]
            prompt_text = self._render_prompt(base_messages, add_generation_prompt=False)
            assistant_header = (
                f"<|start_header_id|>{self.assistant_alias}<|end_header_id|>\n\n{prefix}"
                if prefix
                else f"<|start_header_id|>{self.assistant_alias}<|end_header_id|>\n\n"
            )
            if prompt_text and not prompt_text.endswith('\n'):
                prompt_text += '\n'
            prompt_text += assistant_header
        else:
            prompt_text = self._build_prompt()

        self.queue.put(('assistant_begin', continuation_entry))
        self.queue.put(('streaming_state', True))

        try:
            for chunk in stream_completions(
                prompt_text,
                base_url=self.base_url,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                stop_sequences=stop_sequences,
                seed=self.seed,
                stop_event=self.stop_event,
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

                audio_player = self._collect_audio_player()
                if audio_player is None:
                    continue

                cumulative = ''.join(assistant_chunks)
                for segment, new_index in extract_speakable_segments(cumulative, spoken_upto):
                    try:
                        audio_player.speak_text(segment, voice=self.voice, speed=self.speed)
                    except Exception as exc:
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
                    assistant_entry = {'role': 'assistant', 'name': self.assistant_alias, 'content': assistant_reply}
                    self.messages.append(assistant_entry)
            if assistant_reply and not self.text_only and spoken_upto < len(assistant_reply):
                remaining = assistant_reply[spoken_upto:]
                if remaining.strip():
                    audio_player = self._collect_audio_player()
                    if audio_player is not None:
                        try:
                            audio_player.speak_text(remaining, voice=self.voice, speed=self.speed)
                        except Exception as exc:  # pragma: no cover - defensive
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

    def display_system_message(self, text: str, message_entry: Optional[Dict[str, Any]] = None) -> tk.Frame:
        container, bubble, label, name_label = self._create_message_bubble(
            'system',
            text,
            italic=True,
            message_entry=message_entry,
        )
        if message_entry is not None:
            key = id(message_entry)
            self.message_container_map[key] = container
            self.message_bubble_map[key] = bubble
            self.message_label_map[key] = label
            self.message_name_map[key] = name_label
        self._scroll_to_bottom()
        return container

    def display_user_message(self, message_entry: Dict[str, Any]) -> None:
        self._create_message_bubble('user', message_entry.get('content', ''), message_entry=message_entry)

    def start_assistant_message(self, message_entry: Optional[Dict[str, Any]] = None) -> None:
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

    def finalize_assistant_message(self, payload: Tuple[str, Optional[Dict[str, Any]]] | str) -> None:
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
            self.current_assistant_name_label = None
            return

        if not text.strip():
            if label in self.message_labels:
                self.message_labels.remove(label)
            try:
                container.destroy()
            except Exception:
                pass
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
        self.current_assistant_name_label = None
        self.current_assistant_text = []
        self._scroll_to_bottom()

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
                if self.continuation_notice and self.continuation_notice.winfo_exists():
                    try:
                        self.continuation_notice.destroy()
                    except Exception:
                        pass
                self.continuation_notice = None

        self.process_queue_job = self.root.after(80, self.process_queue)

    def on_message_right_click(self, event: tk.Event, message_entry: Dict[str, Any]) -> None:
        if self.edit_mode_entry is not None:
            return
        if self.streaming_thread and self.streaming_thread.is_alive():
            return
        if message_entry is None:
            return
        if message_entry.get('role') not in {'user', 'assistant', 'system'}:
            return
        if self.context_menu is None:
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
        self.context_menu_target = None
        if not message_entry:
            return

        if not any(entry is message_entry for entry in self.messages):
            messagebox.showinfo('Edit Message', 'This message is no longer part of the active conversation.')
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

    def _continue_current_message(self) -> None:
        message_entry = self.context_menu_target
        self.context_menu_target = None

        if not message_entry or self.edit_mode_entry is not None:
            return
        if self.streaming_thread and self.streaming_thread.is_alive():
            return
        if message_entry.get('role') != 'assistant':
            messagebox.showinfo('Continue Response', 'You can only continue assistant messages.')
            return
        if not self.messages or self.messages[-1] is not message_entry:
            messagebox.showinfo('Continue Response', 'Only the latest assistant reply can be continued.')
            return

        message_entry.setdefault('name', self.assistant_alias)

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
            self.preset = replace(self.preset, system_prompt='')
            self.system_prompt_text.delete('1.0', tk.END)
            self._ensure_system_message(display_if_new=False)

        removed = False
        for idx, existing in enumerate(self.messages):
            if existing is entry:
                self.messages.pop(idx)
                removed = True
                break

        self._remove_message_ui(entry)

        if removed:
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

        if entry is self.system_message_entry:
            self.preset = replace(self.preset, system_prompt=new_text)
            self.system_prompt_text.delete('1.0', tk.END)
            if new_text:
                self.system_prompt_text.insert(tk.END, new_text)

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
        if self._settings_scroll_handlers:
            self._toggle_settings_scroll(False, self._settings_scroll_handlers)

        if self.streaming_thread and self.streaming_thread.is_alive():
            self.stop_event.set()
            self.queue.put(('status', 'Stopping response...'))
            self.queue.put(('streaming_state', False))
            self.streaming_thread.join(timeout=2.0)
            self.streaming_thread = None

        try:
            AudioPlayer.stop()
        except Exception:
            pass

        if self.mix_window is not None and self.mix_window.winfo_exists():
            try:
                self.mix_window.destroy()
            except Exception:
                pass
            self.mix_window = None

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
