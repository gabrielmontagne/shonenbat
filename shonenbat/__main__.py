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

token_to_role = {
    'Q>>': 'user',
    'A>>': 'assistant',
    'S>>': 'system'
}


def focus_prompt(prompt):
    pre = ""
    post = ""

    start_split = re.split(r'^(\s*__START__\s*)$', prompt, flags=re.MULTILINE, maxsplit=1)

    if len(start_split) == 3:
        head, separator, prompt = start_split
        pre = head + separator

    end_split = re.split(r'^(\s*__END__\s*)$', prompt, flags=re.MULTILINE, maxsplit=1)

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
                'content': [
                    { 'type': 'text', 'text': v.strip() }
                ]
            }
        )

    return messages


def run_completion(model, num_options, temperature, full_prompt, max_tokens=4000, instruction=None, stop=[]):
    unescaped_stops = [i.replace('\\n', '\n') for i in stop]

    prompt, pre, post = focus_prompt(full_prompt)

    completion = ''

    if pre:
        completion += pre + '\n'

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
            completion += '{{' + traceback_details + '}}'

    if post:
        completion += '\n' + post

    return completion


def main():
    """Run completion"""

    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType(
        'r'), default=sys.stdin, help='Optionally add {{insert}} for completion')
    parser.add_argument('--max_tokens', '-mt', type=int, default=4000)
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
    parser.add_argument('--model', '-m', type=str, default='dall-e-3')

    args = parser.parse_args()

    prompt, pre, post = focus_prompt(args.prompt.read())

    if pre:
        print(pre)

    try:
        urls = [item['url'] for item in openai.Image.create(
            model=args.model,
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
    parser.add_argument('--max_tokens', '-mt', type=int, default=4096)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    parser.add_argument('--model', '-m', type=str, default='gpt-4-vision-preview')
    parser.add_argument('--output_only', '-oo', action='store_true', help='Skip input echo.')
    parser.add_argument('--offline_preamble', '-op',
                        type=FileType('r'), default=None)
    args = parser.parse_args()
    model = args.model
    num_options = args.num_options
    temperature = args.temperature
    full_prompt = args.prompt.read()
    max_tokens = args.max_tokens
    output_only = args.output_only

    offline_preamble = ''
    if args.offline_preamble:
        offline_preamble = args.offline_preamble.read()

    chat = run_chat(model, num_options, temperature, full_prompt, max_tokens, offline_preamble, output_only)
    print(chat)


def run_chat(model, num_options, temperature, full_prompt, max_tokens=4000, offline_preamble='', output_only=False):

    prompt, pre, post = focus_prompt(full_prompt)

    chat = ''

    if pre:
        chat += pre + '\n'

    try:

        if output_only:
            results = [f'{r.message.content}' for r in openai.ChatCompletion.create(
                model=model,
                messages=messages_from_prompt(offline_preamble + '\n' + prompt),
                n=num_options,
                temperature=temperature,
                max_tokens=max_tokens
            ).choices]
            chat += f'\n\n{"-" * 10}\n\n'.join(results)
        else:
            results = [f'\n\nA>>\n\n{r.message.content}' for r in openai.ChatCompletion.create(
                model=model,
                messages=messages_from_prompt(offline_preamble + '\n' + prompt),
                n=num_options,
                temperature=temperature,
                max_tokens=max_tokens
            ).choices]
            chat += prompt
            chat += f'\n\n{"-" * 10}\n\n'.join(results)
            chat += '\n\nQ>> '

    except Exception as e:
        traceback_details = traceback.format_exc()
        chat += '{{', traceback_details + '}}'

    if post:
        chat += '\n' + post

    return chat

def run_count(model, full_prompt, name=''):
    from tiktoken import encoding_for_model
    prompt, pre, post = focus_prompt(full_prompt)

    count = ''

    if pre:
        count += pre + '\n'

    encoding = encoding_for_model(model)
    tokens = encoding.encode(prompt)

    if name:
        count += f'/* {len(tokens):>10} tokens {name} */'
    else:
        count += f'/* {len(tokens)} */'

    if post:
        count += '\n' + post

    return count


def count():
    parser = ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=FileType(
        'r'), default=sys.stdin, help='Text to count.')

    parser.add_argument('--model', '-m', type=str, default='gpt-4-vision-preview')
    args = parser.parse_args()

    full_prompt = args.prompt.read()
    model = args.model
    name = args.prompt.name
    if name == '<stdin>': name = ''
    count_result = run_count(model, full_prompt, name)
    print(count_result)


def list():
    """Run completion"""

    engines = openai.Engine.list()
    print('list', engines)


if __name__ == '__main__':
    main()
