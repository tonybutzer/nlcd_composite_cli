import subprocess
import socket

def get_ip_address():

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    print(hostname, ip_address)
    return ip_address

def get_awsid(ip_address):
    command = f'aws ec2 describe-instances --filter Name=private-ip-address,Values={ip_address} --query Reservations[].Instances[].InstanceId --output text --region=us-west-2'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Error:", result.stderr)
    awsid = result.stdout.rstrip()
    print(awsid)
    return awsid



def aws_ec2_create_tags(awsid, my_new_tag):
    command = f'aws ec2 create-tags --resources {awsid} --tag \"Key=Name,Value={my_new_tag}\" --region=us-west-2'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Error:", result.stderr)


def tagme(my_new_tag):
    ip_address = get_ip_address()
    awsid=get_awsid(ip_address)
    aws_ec2_create_tags(awsid, my_new_tag)



if __name__ == "__main__":
    my_new_tag='Butzer-HeadNode-V6-pcluster'
    tagme(my_new_tag)

