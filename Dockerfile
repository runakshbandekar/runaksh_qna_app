# pull python base image
FROM python:3.10

# copy application files
ADD . .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["streamlit", "run", "app.py", "--server.port", "8001"]