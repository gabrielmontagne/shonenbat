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
    parser.add_argument('--num_options', '-n', type=int, default=2)
    args = parser.parse_args()
    prompt = args.prompt.read().strip()

    print('max tokens', args.max_tokens)

    print(
        openai.Completion.create(
            engine='davinci',
            prompt=prompt,
            max_tokens=args.max_tokens,
            n=args.num_options
        )
    )

if __name__ == '__main__':
    main()
