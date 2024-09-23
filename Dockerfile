# This sets up the container with Python 3.12 installed.
FROM python:3.12


# Install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev\
    tesseract-ocr-ara\
    ffmpeg\
    libsm6\
    libxext6\
    poppler-utils

# Set the working directory
WORKDIR /app


# Install python dependencies
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt


# This copies everything in your current directory to the /app directory in the container.
COPY . /app


# This command creates a .streamlit directory in the home directory of the container.
RUN mkdir ~/.streamlit


# This copies your Streamlit configuration file into the .streamlit directory you just created.
RUN cp config.toml ~/.streamlit/config.toml


# Similar to the previous step, this copies your Streamlit credentials file into the .streamlit directory.
#RUN cp credentials.toml ~/.streamlit/credentials.toml


# This sets the default command for the container to run the app with Streamlit.
#ENTRYPOINT ["streamlit", "run"]


# This command tells Streamlit to run your app.py script when the container starts.
#CMD ["main_chat_app.py"]

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]