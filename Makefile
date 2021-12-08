NAME := shonenbat
VERSION := 0.0.1

init:
	pip3 install -r requirements.txt

test:
	nosetests

lint:
	@echo '---PEP8---'
	@pep8 -v $(NAME)
	@echo '---pylint---'
	@pylint $(NAME)
