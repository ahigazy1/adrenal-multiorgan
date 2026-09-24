#!/bin/bash
# Create the pseudo-label machine on Google Cloud, or start it again after an interruption. In Google Cloud Shell:
#
#     git clone https://github.com/ahigazy1/adrenal-multiorgan
#     bash adrenal-multiorgan/pseudolabels/gcp/create_vm.sh
#
# Run the second line again whenever the machine stopped before it printed "pseudo-labels complete": it continues.
# Progress:  NAME=adrenal-pseudo bash adrenal-multiorgan/gcp/progress.sh
# When done: NAME=adrenal-pseudo bash adrenal-multiorgan/gcp/delete_vm.sh   (the disk is billed until then)
#
# Same machine as training: g4-standard-48 (RTX PRO 6000 96 GB, 48 vCPUs), Spot, Ubuntu Pro with the NVIDIA driver.
set -euo pipefail
NAME=adrenal-pseudo
DISK_GB=200          # sources about 60 GB, pseudo-labels a few GB
DISK_IOPS=10000      # disk speed as in our test runs; it is a large part of the disk's price
DISK_MB_PER_S=1050
MAX_HOURS=100        # Google stops the machine after this long in one go, whatever it is doing: a ceiling on the bill
IMAGE="image-family=ubuntu-pro-accel-2604-amd64-nvidia-595,image-project=ubuntu-os-accelerator-images"
HERE=$(cd "$(dirname "$0")" && pwd)
command -v cygpath >/dev/null && HERE=$(cygpath -m "$HERE")   # Git Bash on Windows: gcloud needs a Windows-style path
PROJECT=$(gcloud config get-value project 2>/dev/null)
[ -n "$PROJECT" ] || { echo "No project selected. Choose one in the console, or run: gcloud config set project YOUR_PROJECT"; exit 1; }
echo "Project: $PROJECT"
gcloud services enable compute.googleapis.com secretmanager.googleapis.com logging.googleapis.com

# Already created? Then just start it. Its disk lives in one zone, so it can only start there.
ZONE=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
if [ -n "$ZONE" ]; then
    echo "$NAME exists in $ZONE; refreshing its startup script and starting it."
    gcloud compute instances add-metadata "$NAME" --zone="$ZONE" --metadata-from-file=startup-script="$HERE/startup.sh"
    gcloud compute instances start "$NAME" --zone="$ZONE" \
        || { echo "Could not start it. If the message above mentions resources or capacity, $ZONE has no free machine right now: nothing is lost, run this again in an hour."; exit 1; }
    echo "Started. It continues by itself. Progress: NAME=$NAME bash $HERE/../../gcp/progress.sh"
    exit 0
fi

# The Hugging Face token (write access to ahigazy1/adrenal-multiorgan-cache and
# ahigazy1/adrenal-multiorgan-model) is kept in Secret Manager under the name HF_TOKEN, not on the disk and not in
# this repository. If the secret already exists it is used as it is.
if ! gcloud secrets describe HF_TOKEN >/dev/null 2>&1; then
    read -r -s -p "Paste the Hugging Face token, then press Enter (nothing is shown while you paste): " TOKEN; echo
    printf %s "$TOKEN" | gcloud secrets create HF_TOKEN --data-file=-
    unset TOKEN
fi
ACCOUNT="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding HF_TOKEN --member="serviceAccount:$ACCOUNT" \
    --role=roles/secretmanager.secretAccessor >/dev/null

# Spot GPU capacity differs by zone and by hour, so every us-central1 zone that offers this machine type is tried
# in turn until one has room. Only "no capacity" moves on to the next zone; any other error (quota,
# billing, permissions) would be the same everywhere, so it is shown and the script stops.
ZONES="us-central1-f us-central1-a us-central1-b us-central1-c"
for ZONE in $ZONES; do
    echo "Trying $ZONE ... (up to two minutes, nothing is printed meanwhile)"
    if ERROR=$(gcloud compute instances create "$NAME" --zone="$ZONE" \
        --machine-type=g4-standard-48 --provisioning-model=SPOT --instance-termination-action=STOP --max-run-duration="${MAX_HOURS}h" \
        --create-disk="auto-delete=no,boot=yes,size=$DISK_GB,type=hyperdisk-balanced,provisioned-iops=$DISK_IOPS,provisioned-throughput=$DISK_MB_PER_S,$IMAGE" \
        --network-interface=network=default,nic-type=GVNIC \
        --service-account="$ACCOUNT" --scopes=cloud-platform \
        --metadata=serial-port-logging-enable=true --metadata-from-file=startup-script="$HERE/startup.sh" \
        --labels=workload=adrenal-pseudo 2>&1); then
        echo "Created $NAME in $ZONE. It installs TotalSegmentator, downloads the scans and labels them; it switches itself off when done."
        echo "Progress: NAME=$NAME bash $HERE/../../gcp/progress.sh"
        exit 0
    fi
    if ! grep -qiE "ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|currently unavailable|not available in|does not exist in zone|Invalid value for field .resource.machineType" <<<"$ERROR"; then
        echo "$ERROR"
        echo "This is not a capacity problem, so trying other zones would not help. If the message mentions QUOTA, the"
        echo "project needs GPU quota: console.cloud.google.com/iam-admin/quotas, search for 'RTX PRO 6000' and 'GPUs (all regions)'."
        exit 1
    fi
done
echo "No us-central1 zone had a free machine right now. Try again in an hour."
exit 1
