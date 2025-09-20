#!/usr/bin/env python3
"""
EC2 Training Status Checker
Run this to see what's happening on your EC2 instance during training
"""

import boto3
import os
from config import settings
import asyncio

async def check_ec2_status():
    """Check the status of your EC2 training instance"""

    # Get your EC2 client
    ec2_client = boto3.client('ec2', region_name=settings.AWS_REGION)

    print("=" * 60)
    print("EC2 TRAINING INSTANCE STATUS CHECK")
    print("=" * 60)
    print(f"Instance ID: {settings.EXISTING_GPU_INSTANCE_ID}")
    print(f"Region: {settings.AWS_REGION}")
    print()

    try:
        # Check instance status
        response = ec2_client.describe_instances(
            InstanceIds=[settings.EXISTING_GPU_INSTANCE_ID]
        )

        instance = response['Reservations'][0]['Instances'][0]
        state = instance['State']['Name']

        print(f"Instance State: {state}")

        if state == 'running':
            # Get public IP
            public_ip = instance.get('PublicIpAddress', 'No public IP')
            print(f"Public IP: {public_ip}")

            # Check if Docker is running
            ssm_client = boto3.client('ssm', region_name=settings.AWS_REGION)

            try:
                # Check Docker status
                docker_check = ssm_client.send_command(
                    InstanceIds=[settings.EXISTING_GPU_INSTANCE_ID],
                    DocumentName='AWS-RunShellScript',
                    Parameters={'commands': ['docker ps', 'docker info']}
                )

                print("\nDocker Status:")
                print("- Docker containers running"
                print("- Docker daemon status"
            except Exception as e:
                print(f"Could not check Docker: {e}")

            # Check training directory
            try:
                training_check = ssm_client.send_command(
                    InstanceIds=[settings.EXISTING_GPU_INSTANCE_ID],
                    DocumentName='AWS-RunShellScript',
                    Parameters={'commands': [
                        'ls -la /home/ubuntu/ai-training/',
                        'docker ps | grep ai-training-container',
                        'cat /tmp/training_logs/* 2>/dev/null | tail -20 || echo "No training logs found"'
                    ]}
                )

                print("\nTraining Environment:")
                print("- Directory contents"
                print("- Container status"
                print("- Recent training logs"
            except Exception as e:
                print(f"Could not check training environment: {e}")

        else:
            print(f"❌ Instance is {state} - not ready for training")

    except Exception as e:
        print(f"Error checking instance: {e}")
        print("Make sure your EC2 instance is running and accessible")

if __name__ == "__main__":
    asyncio.run(check_ec2_status())
