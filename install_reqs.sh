#!/bin/bash
cd /home/jordan/MRs/metadata/

# Install a specific version of setuptools
# /home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install setuptools==59.5.0


# Install the current directory as a package
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install -e .

# Install specific versions of accelerate and deepspeed
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install accelerate==0.10.0 deepspeed==0.7.7

# Install xxhash
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install xxhash

# Install REL from a GitHub repository
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install "REL @ git+https://github.com/manandey/REL.git#egg=REL"

# Install bs_dateutil from a specific commit in a GitHub repository
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install "bs_dateutil @ git+https://github.com/cccntu/dateutil.git@2.8.5"

# Install specific versions of wikipedia2vec and nltk
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install wikipedia2vec==1.0.5
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install nltk==3.6.7

# Install rich and htmlmin
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install rich
/home/jordan/miniconda3/envs/jordan_metadata_312/bin/pip3 install htmlmin
