FROM python:3.5.5-slim-jessie

RUN pip3 install --upgrade pip && \
	pip3 install --user numpy scipy matplotlib ipython jupyter pandas sympy nose && \
	pip3 install -U scikit-learn

COPY /titanic.csv /data/

COPY /titanic.py /

ENTRYPOINT ["python", "titanic.py"]