#!/bin/bash
# Delete the training machine AND its disk, so that nothing is billed any more. Run in Google Cloud Shell when the
# work is finished. Checkpoints, logs and predictions that were uploaded to Hugging Face are not affected;
# anything that exists only on the machine's disk is lost for good.
set -euo pipefail
NAME=adrenal-train
ZONE=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
[ -n "$ZONE" ] || { echo "There is no machine called $NAME in this project. Nothing to delete."; exit 0; }
read -r -p "Delete $NAME in $ZONE and its disk for good? Type yes: " ANSWER
[ "$ANSWER" = yes ] || { echo "Nothing deleted."; exit 0; }
gcloud compute instances delete "$NAME" --zone="$ZONE" --delete-disks=all --quiet
echo "Deleted. Remaining disks in this project (should be empty):"
gcloud compute disks list
