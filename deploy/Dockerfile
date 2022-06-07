FROM python:3.10-buster

LABEL maintainer="klaus.eckelt@jku.at"
WORKDIR /phovea

COPY requirements*.txt ./
RUN (apt-get update) && \
    (pip install --no-cache-dir -r requirements.txt) && \
    (pip install --no-cache-dir -r requirements_dev.txt)

# Create entrypoint installing the volume-mounted workspace repos
RUN echo '#!/bin/sh\npip install --no-cache-dir --no-deps -r /phovea/requirements_workspace.txt\n$@' > /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT [ "/tmp/entrypoint.sh" ]

# Default command is to start the server
CMD [ "python -m uvicorn kokiri.server:app --reload --host 0.0.0.0 --port 9666" ]

EXPOSE 9666