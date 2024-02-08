#!/usr/bin/env python
# coding: utf-8

import subprocess
import json
import pandas as pd
from tabulate import tabulate

import boto3
import socket



def get_instance_name(instance_id):
    # Specify the region
    region = 'us-west-2'  # Replace with your AWS region, e.g., 'us-east-1'

    # Create EC2 client
    ec2 = boto3.client('ec2', region_name=region)

    try:
        # Describe the instance to get its tags
        response = ec2.describe_instances(InstanceIds=[instance_id])

        # Extract the "Name" tag value
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                for tag in instance.get('Tags', []):
                    if tag['Key'] == 'Name':
                        return tag['Value']

        # If the instance has no "Name" tag
        return f"No 'Name' tag found for instance {instance_id}"

    except Exception as e:
        return f"Error: {str(e)}"



def describe_instances_to_dataframe():
    # Run AWS CLI command to describe instances
    command = "aws ec2 describe-instances --region=us-west-2"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Parse JSON output
    try:
        instances_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

    # Extract relevant information for DataFrame
    instances_list = []
    for reservation in instances_data.get('Reservations', []):
        for instance in reservation.get('Instances', []):
            instance_info = {
                'InstanceId': instance.get('InstanceId', ''),
                'InstanceType': instance.get('InstanceType', ''),
                'State': instance.get('State', {}).get('Name', ''),
                'PublicIpAddress': instance.get('PublicIpAddress', ''),
                'PrivateIpAddress': instance.get('PrivateIpAddress', ''),
                # Add more fields as needed
            }
            instances_list.append(instance_info)

    # Create Pandas DataFrame
    df = pd.DataFrame(instances_list)
    df['InstanceName'] = df['InstanceId'].apply(get_instance_name)
    df = df[['InstanceId','InstanceType','State','PrivateIpAddress','InstanceName']]

    return df



#df = describe_instances_to_dataframe()



#selected_rows = df[df['InstanceName'].isin(['HeadNode', 'Compute'])]


#selected_rows

df = describe_instances_to_dataframe()


df = df[df['State'].isin(['running', ])]
#print(tabulate(df, headers='keys', tablefmt='fancy_grid'))



selected_rows = df[df['InstanceName'].str.startswith('Compute')]

print(tabulate(selected_rows, headers='keys', tablefmt='fancy_grid'))






