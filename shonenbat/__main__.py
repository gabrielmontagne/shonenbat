from argparse import ArgumentParser, FileType
from dotenv import load_dotenv
import sys
import openai
import os


def main():
    """Run completion"""

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType('r'), default=sys.stdin)
    parser.add_argument('--max_tokens', '-mt', type=int, default=50)
    parser.add_argument('--num_options', '-n', type=int, default=3)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    args = parser.parse_args()
    prompt = args.prompt.read().strip()

    results = [r.text for r in openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=args.max_tokens,
        echo=True,
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
