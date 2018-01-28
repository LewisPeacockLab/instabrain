FROM neurodebian:stretch-non-free

ENV DEBIAN_FRONTEND=noninteractive
ENV AFNI_INSTALLDIR=/usr/lib/afni
ENV PATH="${PATH}:${AFNI_INSTALLDIR}/bin" \
  AFNI_PLUGINPATH="${AFNI_INSTALLDIR}/plugins" \
  AFNI_MODELPATH="${AFNI_INSTALLDIR}/models" \
  AFNI_TTATLAS_DATASET=/usr/share/afni/atlases \
  AFNI_IMSAVE_WARNINGS=NO

RUN echo yes | apt-get -qy update \
  && apt-get -qy install python-pip python-setuptools afni python-mvpa2 --no-install-recommends \
  && apt-get clean

WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /usr/src/app

WORKDIR /usr/src/app/realtime
CMD ["/usr/bin/python", "instabrain.py", "-s", "ft001"]
