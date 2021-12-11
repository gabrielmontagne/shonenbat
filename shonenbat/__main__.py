from dotenv import load_dotenv
import os


def main():
    """do it"""

    load_dotenv()

    print('running module shonenbat', os.getenv('OPENAI_ORG_ID'))


if __name__ == '__main__':
    main()
