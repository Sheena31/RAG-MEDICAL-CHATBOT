# Use the official Jenkins LTS image as base
FROM jenkins/jenkins:lts

# Switch to root to install dependencies
USER root

# Install prerequisites, Docker CE CLI, and AWS CLI
RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common awscli && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure docker group exists and add Jenkins user
RUN groupadd -f docker && \
    usermod -aG docker jenkins

# Mount Docker socket from host so Jenkins can talk to host Docker
VOLUME /var/run/docker.sock

# (Optional) If you want DinD, keep Docker data directory
VOLUME /var/lib/docker

# Switch back to Jenkins user
USER jenkins
