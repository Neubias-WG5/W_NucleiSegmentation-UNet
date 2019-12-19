FROM python:3.6.9-stretch

# ------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.3.0.poc.1 && pip install . && \
    rm -r /Cytomine-python-client

# ------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git && \
    cd /neubiaswg5-utilities/ && git checkout tags/v0.8.6 && pip install .

# install utilities binaries
RUN chmod +x /neubiaswg5-utilities/bin/*
RUN cp /neubiaswg5-utilities/bin/* /usr/bin/ && \
    rm -r /neubiaswg5-utilities

# ------------------------------------------------------------------------------

RUN pip install keras
RUN pip install tensorflow

RUN git clone https://github.com/carpenterlab/2019_caicedo_cytometryA.git

RUN mkdir /app && mkdir /app/utils
RUN cp /2019_caicedo_cytometryA/unet4nuclei/utils/* /app/utils/
ADD model_builder.py /app/utils/model_builder.py

ADD unet_weights.h5 /app/weights.h5
RUN chmod 444 /app/weights.h5

ADD unet_utils.py /app/unet_utils.py
ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python3.6","/app/wrapper.py"]
