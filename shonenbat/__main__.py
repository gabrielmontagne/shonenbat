from dotenv import load_dotenv
import openai
import os


def main():
    """Run completion"""

    load_dotenv()
    openai.organization = os.getenv("OPENAI_ORG_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(openai.Engine.list())

if __name__ == '__main__':
    main()
