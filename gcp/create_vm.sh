#!/bin/bash
# Create the training machine on Google Cloud, or start it again after an interruption.
#
# Where to run it: Google Cloud Shell (console.cloud.google.com, the ">_" button at the top right). Nothing to
# install. Pick your project in the console first, then paste these two lines:
#
#     git clone https://github.com/ahigazy1/adrenal-multiorgan
#     bash adrenal-multiorgan/gcp/create_vm.sh
#
# Run the second line again whenever the machine has stopped before training finished: it starts the existing
# machine, which continues from the last checkpoint by itself (see startup.sh).
#
# The machine: g4-standard-48 (one NVIDIA RTX PRO 6000 96 GB, 48 vCPUs, 180 GB RAM), Spot pricing, Ubuntu Pro with
# the NVIDIA driver preinstalled, one 500 GB disk. Google can stop a Spot machine at any time; nothing is lost.
# The disk is billed while it exists, even when the machine is stopped: run delete_vm.sh when the work is done.
set -euo pipefail
NAME=adrenal-train
DISK_GB=500          # the training cache is about 200 GB
DISK_IOPS=10000      # disk speed as in our test runs; it is a large part of the disk's price
DISK_MB_PER_S=1050
IMAGE=projects/ubuntu-os-accelerator-images/global/images/ubuntu-pro-accel-2604-amd64-nvidia-595-v20260909
HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT=$(gcloud config get-value project 2>/dev/null)
[ -n "$PROJECT" ] || { echo "No project selected. Choose one in the console, or run: gcloud config set project YOUR_PROJECT"; exit 1; }
echo "Project: $PROJECT"

# Already created? Then just start it.
ZONE=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
if [ -n "$ZONE" ]; then
    echo "$NAME exists in $ZONE; starting it."
    gcloud compute instances start "$NAME" --zone="$ZONE"
    echo "Started. Training continues by itself. Progress: bash $HERE/progress.sh"
    exit 0
fi

gcloud services enable compute.googleapis.com secretmanager.googleapis.com

# The Hugging Face token (needs read access to the cache and write access to the model repository) is kept in
# Secret Manager, not on the disk and not in this repository.
if ! gcloud secrets describe HF_TOKEN >/dev/null 2>&1; then
    read -r -s -p "Paste the Hugging Face token, then press Enter (nothing is shown while you paste): " TOKEN; echo
    printf %s "$TOKEN" | gcloud secrets create HF_TOKEN --data-file=-
    unset TOKEN
fi
ACCOUNT="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding HF_TOKEN --member="serviceAccount:$ACCOUNT" \
    --role=roles/secretmanager.secretAccessor >/dev/null

# Google publishes no "least busy zone". Spot GPU capacity differs by zone and by hour, so every zone that offers
# this machine type is tried in turn (United States first) until one has room.
ALL_ZONES=$(gcloud compute machine-types list --filter="name=g4-standard-48" --format="value(zone)" | sort)
ZONES="$(grep '^us-' <<<"$ALL_ZONES" || true) $(grep -v '^us-' <<<"$ALL_ZONES" || true)"
for ZONE in $ZONES; do
    echo "Trying $ZONE ..."
    if gcloud beta compute instances create "$NAME" --zone="$ZONE" \
        --machine-type=g4-standard-48 --provisioning-model=SPOT --instance-termination-action=STOP \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --create-disk="auto-delete=no,boot=yes,size=$DISK_GB,type=hyperdisk-balanced,provisioned-iops=$DISK_IOPS,provisioned-throughput=$DISK_MB_PER_S,image=$IMAGE" \
        --network-interface=network=default,nic-type=GVNIC \
        --service-account="$ACCOUNT" --scopes=cloud-platform \
        --metadata-from-file=startup-script="$HERE/startup.sh" --labels=workload=adrenal-multiorgan; then
        echo "Created $NAME in $ZONE. It sets itself up and starts training; the first start takes about 30 minutes."
        echo "Progress: bash $HERE/progress.sh"
        exit 0
    fi
done
echo "No zone had a free machine right now. Try again in an hour."
exit 1
