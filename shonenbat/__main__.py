from argparse import ArgumentParser, FileType
from dotenv import load_dotenv
import traceback
import sys
import openai
import os
import subprocess
import re

split_token = '{{insert}}'

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def messages_from_prompt(prompt):
    token = r'(?:^|\n)(\w>>)'
    preamble, *pairs = re.split(token, prompt)
    qa = [pair for pair in (zip(pairs[::2], pairs[1::2])) if pair[1]]

    preamble = preamble.strip()

    messages = []

    if preamble:
        messages.append(
            {
                'role': 'system', 
                'content': preamble
            }
        )

    for k, v in qa:
        print(k, v.strip())
        if k == 'Q>>':
            messages.append(
                {
                    'role': 'user', 
                    'content': v.strip()
                }
            )

        if k == 'A>>':
            messages.append(
                {
                    'role': 'assistant', 
                    'content': v.strip()
                }
            )

    return messages


def main():
    """Run completion"""

    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType(
        'r'), default=sys.stdin, help='Optionally add {{insert}} for completion')
    parser.add_argument('--max_tokens', '-mt', type=int, default=1000)
    parser.add_argument('--num_options', '-n', type=int, default=1)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    parser.add_argument('--model', '-m', type=str, default='text-davinci-003')
    parser.add_argument('--instruction', '-i', type=str, default=None)
    args = parser.parse_args()
    prompt = args.prompt.read().strip()
    instruction = args.instruction
    suffix = None

    if split_token in prompt:
        prompt, suffix = prompt.split(split_token, maxsplit=1)

    if suffix:
        results = [f'{prompt}{r.text}{suffix}' for r in openai.Completion.create(
            engine=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n=args.num_options,
            suffix=suffix
        ).choices]
        print(f'\n\n{"█" * 10}\n\n'.join(results))
    elif instruction:
        results = [f'{r.text}' for r in openai.Edit.create(
            engine='text-davinci-edit-001',
            input=prompt,
            instruction=instruction,
            temperature=args.temperature,
            n=args.num_options,
        ).choices]
        print(f'\n\n{"×" * 10}\n\n'.join(results))
        print(f'\n\n{"·" * 10}\n\n[{instruction}]')

    else:
        try:
            results = [f'{prompt}{r.text}' for r in openai.Completion.create(
                engine=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                n=args.num_options
            ).choices]
            print(f'\n\n{"█" * 10}\n\n'.join(results))
        except Exception as e:
            traceback_details = traceback.format_exc()
            print('{{', traceback_details + '}}')


def image():
    """Run image generation"""

    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType(
        'r'), default=sys.stdin, help='Image description')
    parser.add_argument('--num_options', '-n', type=int, default=1)
    parser.add_argument('--size', '-s', type=str, default='1024x1024')
    parser.add_argument('--command', '-c', type=str,
                        help='Optional command to run for each generated URL')

    args = parser.parse_args()
    prompt = args.prompt.read().strip()

    try:
        urls = [item['url'] for item in openai.Image.create(
            prompt=prompt,
            n=args.num_options,
            size=args.size
        )['data']]

        print(f'{prompt}\n\n')
        print(f'\n\n{"█" * 10}\n\n'.join(urls))

        if args.command:
            for url in urls:
                subprocess.run([args.command, url])

    except Exception as e:
        traceback_details = traceback.format_exc()
        print('{{', traceback_details + '}}')


def chat():

    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType(
        'r'), default=sys.stdin, help='Optionally add {{insert}} for completion')
    parser.add_argument('--max_tokens', '-mt', type=int, default=1000)
    parser.add_argument('--num_options', '-n', type=int, default=1)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    parser.add_argument('--model', '-m', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()
    prompt = args.prompt.read().strip()

    try:
        results = [f'{r.message.content}' for r in openai.ChatCompletion.create(
            model=args.model,
            messages=messages_from_prompt(prompt)
        ).choices]

        print(results)

    except Exception as e:
        traceback_details = traceback.format_exc()
        print('{{', traceback_details + '}}')


def list():
    """Run completion"""

    engines = openai.Engine.list()
    print('list', engines)


if __name__ == '__main__':
    main()
