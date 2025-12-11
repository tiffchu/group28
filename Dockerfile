# Use Miniforge as base
FROM condaforge/miniforge3:25.11.0-0

# Install system dependencies (make)
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# copy the lockfile into the container
COPY conda-lock.yml conda-lock.yml

# Install conda-lock in base environment
RUN conda install -n base -c conda-forge conda-lock -y

# Create dockerlock environment from lockfile
RUN conda-lock install -n dockerlock conda-lock.yml

# make dockerlock the default environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate dockerlock" >> ~/.bashrc

# set the default shell to use bash with login to pick up bashrc
# this ensures that we are starting from an activated dockerlock environment
SHELL ["/bin/bash", "-l", "-c"]

# expose JupyterLab port
EXPOSE 8888

# sets the default working directory
# this is also specified in the compose file
WORKDIR /workspace

# run JupyterLab on container start
# uses the jupyterlab from the install environment
CMD ["conda", "run", "--no-capture-output", "-n", "dockerlock", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]
