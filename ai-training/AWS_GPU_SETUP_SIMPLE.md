# 🚀 AWS GPU Training Setup - Simple Step-by-Step Guide

**Yes, you CAN see training logs in real-time!** The GPU training supports live progress tracking just like CPU training.

## 🎯 **What You'll Get**
- **10-50x faster training** (15-30 minutes instead of 2-4 hours)
- **Real-time training logs** with progress updates
- **Professional NVIDIA GPUs** (T4 or V100)
- **Cost: ~$0.13-0.26 per training session**
- **Automatic cleanup** - instances terminate after training

---

## 📋 **Step 1: AWS Account Setup (5 minutes)**

### 1.1 Login to AWS Console
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Sign in to your account
3. Make sure billing is enabled (you'll see a credit card on file)

### 1.2 Get Your AWS Credentials
1. Click your username (top right) → **"Security credentials"**
2. Click **"Create access key"**
3. Choose **"Command Line Interface (CLI)"**
4. **SAVE THESE CREDENTIALS** - you'll need them:
   - Access Key ID: `AKIA...`
   - Secret Access Key: `...`

---

## 🔧 **Step 2: Install AWS CLI (2 minutes)**

### On Windows:
```bash
# Download and run the installer from:
# https://awscli.amazonaws.com/AWSCLIV2.msi
```

### On Mac:
```bash
brew install awscli
```

### On Linux:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Configure AWS CLI:
```bash
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key  
# Enter region: us-east-1
# Enter output format: json
```

---

## 🛠️ **Step 3: Create Required AWS Resources (10 minutes)**

### 3.1 Create S3 Bucket
```bash
# Replace 'your-unique-bucket-name' with something unique
aws s3 mb s3://your-unique-bucket-name --region us-east-1
```

### 3.2 Create IAM Role (Copy & Paste This)
```bash
# Create the role
aws iam create-role --role-name EC2-S3-Access --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach S3 permissions
aws iam attach-role-policy --role-name EC2-S3-Access --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-S3-Access
aws iam add-role-to-instance-profile --instance-profile-name EC2-S3-Access --role-name EC2-S3-Access
```

### 3.3 Create Security Group
```bash
# Create security group
aws ec2 create-security-group --group-name gpu-training-sg --description "Security group for GPU training instances"

# Get your IP address (replace 0.0.0.0/0 with your actual IP for security)
# Visit: https://whatismyipaddress.com/
aws ec2 authorize-security-group-ingress --group-name gpu-training-sg --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name gpu-training-sg --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name gpu-training-sg --protocol tcp --port 443 --cidr 0.0.0.0/0
```

### 3.4 Create Key Pair
```bash
# Create key pair for SSH access
aws ec2 create-key-pair --key-name gpu-training-key --query 'KeyMaterial' --output text > gpu-training-key.pem
chmod 400 gpu-training-key.pem
```

### 3.5 Get Your Subnet ID
```bash
# Get your default subnet ID
aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text
```

---

## ⚙️ **Step 4: Configure Your App (2 minutes)**

### 4.1 Update Your .env File
Add these lines to your `.env` file:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET=your-unique-bucket-name

# GPU Training Configuration
AWS_KEY_NAME=gpu-training-key
AWS_SECURITY_GROUP_ID=sg-xxxxxxxxx  # From security group creation above
AWS_SUBNET_ID=subnet-xxxxxxxxx      # From subnet command above
```

### 4.2 Get Your Resource IDs
Run these commands to get the IDs you need:

```bash
# Get Security Group ID
aws ec2 describe-security-groups --group-names gpu-training-sg --query 'SecurityGroups[0].GroupId' --output text

# Get Subnet ID  
aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text
```

---

## 🚀 **Step 5: Test GPU Training (5 minutes)**

### 5.1 Start Your App
```bash
cd /workspace/ai-training
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5.2 Test GPU Training
1. Go to your frontend
2. Upload some signature images
3. Click **"Train Model"** 
4. **You'll see real-time training logs!** 📊

---

## 📊 **Real-Time Training Logs - How It Works**

### ✅ **You WILL See:**
- **Live progress updates** (0% → 100%)
- **Current training stage** (preprocessing → training → saving)
- **Epoch progress** (Epoch 1/50, 2/50, etc.)
- **Loss and accuracy** in real-time
- **Estimated time remaining**
- **Training completion status**

### 📱 **Where You See Logs:**
- **Frontend Progress Bar** - Visual progress indicator
- **Training Logs Panel** - Detailed text logs
- **Real-time Updates** - Live streaming of training metrics

---

## 💰 **Cost Breakdown**

| Instance Type | GPU | Cost/Hour | Training Time | Total Cost |
|---------------|-----|-----------|---------------|------------|
| g4dn.xlarge   | 1x T4 | $0.526 | ~15-30 min | $0.13-0.26 |
| g4dn.2xlarge  | 1x T4 | $0.752 | ~10-20 min | $0.13-0.25 |
| p3.2xlarge    | 1x V100 | $3.06 | ~5-10 min | $0.26-0.51 |

**Recommendation**: Start with `g4dn.xlarge` - best value for money!

---

## 🔍 **Monitor Your Training**

### Check Instance Status:
```bash
# See running GPU instances
aws ec2 describe-instances --filters "Name=tag:Purpose,Values=AI-Training" --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' --output table
```

### View Training Logs:
```bash
# SSH into instance (if needed for debugging)
ssh -i gpu-training-key.pem ubuntu@<instance-ip>

# Check training logs
tail -f /var/log/cloud-init-output.log
```

---

## 🛠️ **Troubleshooting**

### ❌ **Common Issues & Solutions:**

1. **"Access Denied" Error**
   - Check your AWS credentials in `.env`
   - Verify IAM role was created correctly

2. **"Instance Launch Failed"**
   - Check security group ID in `.env`
   - Verify subnet ID is correct

3. **"No Training Logs"**
   - Check your frontend is connected to the right API
   - Verify job ID is being passed correctly

4. **"High Costs"**
   - Set up billing alerts in AWS Console
   - Use smaller instance types
   - Monitor instance termination

### 💡 **Pro Tips:**
- **Set up billing alerts** in AWS Console (Billing → Budgets)
- **Use Spot Instances** for 90% cost savings (advanced)
- **Monitor costs** with AWS Cost Explorer
- **Test with small datasets** first

---

## 🎯 **Quick Start Checklist**

- [ ] AWS account with billing enabled
- [ ] AWS CLI installed and configured
- [ ] S3 bucket created
- [ ] IAM role created
- [ ] Security group created
- [ ] Key pair created
- [ ] Subnet ID obtained
- [ ] `.env` file updated with all IDs
- [ ] App restarted with new config
- [ ] Test training with sample data

---

## 🚀 **You're Ready!**

Once you complete these steps, your GPU training will be:
- **10-50x faster** than CPU training
- **Show real-time logs** just like CPU training
- **Cost only $0.13-0.26 per session**
- **Automatically clean up** after training

**Start with Step 1 and work through each step - it's easier than it looks!** 🎉

---

## 📞 **Need Help?**

If you get stuck on any step:
1. **Check the error message** - it usually tells you what's wrong
2. **Verify your `.env` file** has all the correct IDs
3. **Test AWS CLI** with `aws s3 ls` (should list your bucket)
4. **Check AWS Console** to see if resources were created

**Your AI training is about to get 50x faster!** 🚀