FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE rhizotron.settings.prod

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

RUN pip install poetry gunicorn

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8000/tcp

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "rhizotron.wsgi:application"]
