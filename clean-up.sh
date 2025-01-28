#!/bin/bash

# Display current disk usage
echo "Current disk usage:"
df -h

# Stop all running Docker containers
echo "Stopping all running Docker containers..."
docker stop $(docker ps -q)

# Remove all stopped Docker containers
echo "Removing all stopped Docker containers..."
docker rm $(docker ps -a -q)

# Remove all unused Docker images (dangling and unreferenced)
echo "Removing all unused Docker images..."
docker rmi $(docker images -q -f "dangling=true")

# Remove all unused volumes
echo "Removing all unused Docker volumes..."
docker volume prune -f

# Remove all unused networks
echo "Removing all unused Docker networks..."
docker network prune -f

# Clean up unused build cache
echo "Cleaning up Docker build cache..."
docker builder prune -f

# Remove all Docker images (optional, be cautious with this step)
# Uncomment the next line to remove all Docker images
# echo "Removing all Docker images..."
# docker rmi $(docker images -q)

# Remove all Docker containers (optional, be cautious with this step)
# Uncomment the next line to remove all Docker containers
# echo "Removing all Docker containers..."
# docker rm $(docker ps -a -q)

# Display disk usage after cleanup
echo "Disk usage after cleanup:"
df -h

# Optional: Restart Docker service (for some systems it might help with resources)
# echo "Restarting Docker service..."
# sudo systemctl restart docker