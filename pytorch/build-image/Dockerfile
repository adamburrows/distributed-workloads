FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN mkdir -p /opt/pytorch-mnist
ADD ./ /opt/pytorch-mnist
WORKDIR /opt/pytorch-mnist

# Add folder for the logs.
RUN mkdir /katib

RUN chgrp -R 0 /opt/pytorch-mnist \
  && chmod -R g+rwX /opt/pytorch-mnist \
  && chgrp -R 0 /katib \
  && chmod -R g+rwX /katib

ENTRYPOINT ["python3", "/opt/pytorch-mnist/mnist.py"]
