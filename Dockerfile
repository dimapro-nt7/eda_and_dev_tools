FROM python:3.10-slim-buster

RUN pip install explainerdashboard

COPY app/eda.py ./
COPY app/model.py ./
COPY app/dashboard.py ./
COPY app/app.py ./

RUN python eda.py
RUN python model.py
RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]
