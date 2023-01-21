from argparse import ArgumentParser, FileType
from dotenv import load_dotenv
import sys
import openai
import os

split_token = '{{insert}}'

def main():
    """Run completion"""

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType('r'), default=sys.stdin, help='Optionally add {{insert}} for completion')
    parser.add_argument('--max_tokens', '-mt', type=int, default=3000)
    parser.add_argument('--num_options', '-n', type=int, default=3)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    parser.add_argument('--instruction', '-i', type=str, default=None)
    args = parser.parse_args()
    prompt = args.prompt.read().strip()
    instruction = args.instruction
    suffix = None

    if split_token in prompt:
        prompt, suffix = prompt.split(split_token, maxsplit=1)

    if suffix:
        results = [f'{prompt}{r.text}{suffix}' for r in openai.Completion.create(
            engine='text-davinci-003',
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
        results = [f'{prompt}{r.text}' for r in openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=args.max_tokens,
            # echo=True,
            temperature=args.temperature,
            n=args.num_options
        ).choices]
        print(f'\n\n{"▒" * 10}\n\n'.join(results))


def list():
    """Run completion"""

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    engines = openai.Engine.list()
    print('list', engines)

if __name__ == '__main__':
    main()
