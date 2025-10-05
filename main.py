from __future__ import annotations

import argparse

from tts_chat.app import ChatApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Chat with a local LM Studio server and hear responses in a window.')
    parser.add_argument('-m', '--model', default='lmstudio', help='Model identifier exposed by LM Studio.')
    parser.add_argument('--base-url', default='http://127.0.0.1:1234', help='Base URL where LM Studio serves its API.')
    parser.add_argument('--system', default=None, help='System prompt for the assistant.')
    parser.add_argument('--user-role', dest='user_role', default=None, help='Display name and API role alias for the user.')
    parser.add_argument('--assistant-role', dest='assistant_role', default=None, help='Display name and API role alias for the assistant.')
    parser.add_argument('-v', '--voice', default='af_bella', help='Voice used for TTS playback.')
    parser.add_argument('-s', '--speed', type=float, default=1.0, help='Playback speed for synthesized speech.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for the LLM.')
    parser.add_argument('--max-tokens', type=int, default=0, help='Optional cap on generated tokens (0 = unlimited).')
    parser.add_argument('--timeout', type=float, default=120.0, help='HTTP timeout when querying LM Studio.')
    parser.add_argument('--text-only', action='store_true', help='Disable audio playback and only display responses.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ChatApp(args)
    app.run()


if __name__ == '__main__':
    main()
