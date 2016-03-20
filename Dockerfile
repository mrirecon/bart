FROM ubuntu
RUN apt-get update
RUN apt-get -yy install make gcc libfftw3-dev liblapack-dev libpng-dev
RUN apt-get -yy install wget
RUN wget https://github.com/mrirecon/bart/archive/v0.3.00.tar.gz
RUN tar xzvf v0.3.00.tar.gz
RUN cd bart-0.3.00 && make && make install

