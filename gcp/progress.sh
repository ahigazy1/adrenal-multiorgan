#!/bin/bash
# Show whether the training machine is running and the last lines of its log. Run in Google Cloud Shell.
NAME=adrenal-train
ZONE=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
[ -n "$ZONE" ] || { echo "There is no machine called $NAME in this project."; exit 1; }
STATUS=$(gcloud compute instances describe "$NAME" --zone="$ZONE" --format="value(status)")
echo "$NAME in $ZONE is $STATUS"
if [ "$STATUS" = RUNNING ]; then
    gcloud compute ssh "$NAME" --zone="$ZONE" --command="grep -a 'Epoch [0-9]\|Pseudo dice\|Uploaded\|WARNING\|Error\|=== ' /var/log/adrenal.log | tail -n 12"
else
    echo "It is switched off. If training has not finished, start it again with: bash $(dirname "$0")/create_vm.sh"
    echo "Checkpoints so far: https://huggingface.co/ahigazy1/adrenal-multiorgan-model"
fi
