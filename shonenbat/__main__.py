from argparse import ArgumentParser, FileType
from dotenv import load_dotenv, find_dotenv
import traceback
import sys
import openai
import os
import subprocess
import re

split_token = '{{insert}}'

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

token_to_role = {
    'Q>>': 'user',
    'A>>': 'assistant',
    'S>>': 'system'
}


def focus_prompt(prompt):
    pre = ""
    post = ""

    start_split = re.split(r'^(\s*__START__\s*)$', prompt, flags=re.MULTILINE)

    if len(start_split) == 3:
        head, separator, prompt = start_split
        pre = head + separator

    end_split = re.split(r'^(\s*__END__\s*)$', prompt, flags=re.MULTILINE)

    if len(end_split) == 3:
        prompt, separator, tail = end_split
        post = separator + tail

    return prompt.strip(), pre, post


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
        messages.append(
            {
                'role': token_to_role.get(k, 'S>>'),
                'content': v.strip()
            }
        )

    return messages


def run_completion(model, num_options, temperature, full_prompt, max_tokens=1000, instruction=None, stop=[]):
    unescaped_stops = [i.replace('\\n', '\n') for i in stop]

    prompt, pre, post = focus_prompt(full_prompt)

    completion = ''

    if pre:
        completion += pre

    suffix = None

    if split_token in prompt:
        prompt, suffix = prompt.split(split_token, maxsplit=1)

    if suffix:
        results = [f'{prompt}{r.text}{suffix}' for r in openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=num_options,
            stop=unescaped_stops or None,
            suffix=suffix
        ).choices]
        completion += f'\n\n{"=" * 10}\n\n'.join(results)
    elif instruction:
        results = [f'{r.text}' for r in openai.Edit.create(
            engine='text-davinci-edit-001',
            input=prompt,
            instruction=instruction,
            temperature=temperature,
            n=num_options,
            stop=unescaped_stops or None,
        ).choices]
        completion += f'\n\n{"×" * 10}\n\n'.join(results)
        completion += f'\n\n{"·" * 10}\n\n[{instruction}]'

    else:
        try:
            results = [f'{prompt}{r.text}' for r in openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_options,
                stop=unescaped_stops or None,
            ).choices]
            completion += f'\n\n{"_" * 10}\n\n'.join(results)
        except Exception as e:
            traceback_details = traceback.format_exc()
            completion += '{{', traceback_details + '}}'

    if post:
        completion += post

    return completion


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
    parser.add_argument('--stop', nargs='*', default=[])

    args = parser.parse_args()
    stop = args.stop
    full_prompt = args.prompt.read()
    instruction = args.instruction
    model = args.model
    max_tokens = args.max_tokens
    temperature = args.temperature
    num_options = args.num_options

    completion = run_completion(model, num_options, temperature,
                                full_prompt, max_tokens, instruction, stop)

    print(completion)


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

    prompt, pre, post = focus_prompt(args.prompt.read())

    if pre:
        print(pre)

    try:
        urls = [item['url'] for item in openai.Image.create(
            prompt=prompt,
            n=args.num_options,
            size=args.size
        )['data']]

        print(f'{prompt}\n\n')
        print(f'\n\n{"-" * 10}\n\n'.join(urls))

        if args.command:
            for url in urls:
                subprocess.run([args.command, url])

    except Exception as e:
        traceback_details = traceback.format_exc()
        print('{{', traceback_details + '}}')

    if post:
        print(post)


def chat():

    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType(
        'r'), default=sys.stdin, help='An optional preamble with context for the agent, then block sections headed by Q>> and A>>. ')
    parser.add_argument('--num_options', '-n', type=int, default=1)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    parser.add_argument('--model', '-m', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()
    model = args.model
    num_options = args.num_options
    temperature = args.temperature
    full_prompt = args.prompt.read()

    chat = run_chat(model, num_options, temperature, full_prompt)
    print(chat)


def run_chat(model, num_options, temperature, full_prompt):

    prompt, pre, post = focus_prompt(full_prompt)

    chat = ''

    if pre:
        chat += pre

    try:
        results = [f'\nA>>\n\n{r.message.content}' for r in openai.ChatCompletion.create(
            model=model,
            messages=messages_from_prompt(prompt),
            n=num_options,
            temperature=temperature,

        ).choices]

        chat += prompt
        chat += f'\n\n{"-" * 10}\n\n'.join(results)
        chat += '\nQ>> '

    except Exception as e:
        traceback_details = traceback.format_exc()
        chat += '{{', traceback_details + '}}'

    if post:
        chat += post

    return chat


def list():
    """Run completion"""

    engines = openai.Engine.list()
    print('list', engines)


if __name__ == '__main__':
    main()
