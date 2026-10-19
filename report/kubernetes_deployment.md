# Kubernetes Deployment Guide

## Overview
The Heart Disease Classification API is containerized using Docker and deployed locally using Minikube.

## Components
- Docker image: heart-disease-mlops:latest
- Kubernetes Deployment: heart-disease-api
- Replicas: 1
- Container port: 8000
- Kubernetes Service: heart-disease-api-service
- Service type: NodePort
- NodePort: 30080
- Health endpoint: /health
- Prediction endpoint: /predict

## Prerequisites
- Docker
- Minikube
- kubectl or Minikube's bundled kubectl
- Project source code and trained model files

## Deployment Steps
Run these commands from the project root:

1. Build the image:
   docker build -t heart-disease-mlops:latest .

2. Start Minikube:
   minikube start --driver=docker

3. Load the image:
   minikube image load heart-disease-mlops:latest

4. Apply the manifests:
   minikube kubectl -- apply -f deployment/

5. Check the deployment:
   minikube kubectl -- get deployments

6. Check the pods:
   minikube kubectl -- get pods

7. Check the services:
   minikube kubectl -- get services

## Health Verification
Start port forwarding:

   minikube kubectl -- port-forward service/heart-disease-api-service 8000:8000

In another terminal, run:

   curl http://127.0.0.1:8000/health

Expected response:

   {"status":"healthy"}

## Prediction Verification
Send a JSON request containing all 13 required input features to POST /predict. The API returns the prediction, label, and confidence.

## Health Checks and Resources
The Kubernetes Deployment defines readiness and liveness probes using /health, plus CPU and memory requests and limits.

## Limitations
This is a local Minikube deployment for demonstration and assignment evaluation. It is not a publicly hosted production service.
