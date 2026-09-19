#!/bin/bash
# Show whether the training machine is running and the last lines it printed. Run in Google Cloud Shell.
# The lines come from the machine's serial console, so no login to the machine is needed, and they are
# still available from Cloud Logging after the machine has switched itself off.
NAME=adrenal-train
WANTED='md5sum|OK$|Fetching|cases done|cases written|Preprocessing|Epoch [0-9]|Pseudo dice|Uploaded|WARNING|ERROR|Error|Traceback|=== '
ZONE=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
[ -n "$ZONE" ] || { echo "There is no machine called $NAME in this project."; exit 1; }
STATUS=$(gcloud compute instances describe "$NAME" --zone="$ZONE" --format="value(status)")
echo "$NAME in $ZONE is $STATUS"
if [ "$STATUS" = RUNNING ]; then
    gcloud compute instances get-serial-port-output "$NAME" --zone="$ZONE" 2>/dev/null | grep -a "startup-script:" | grep -av "^\[" | grep -aE "$WANTED" | sed "s/^.*startup-script: //" | tail -n 12
else
    ID=$(gcloud compute instances describe "$NAME" --zone="$ZONE" --format="value(id)")
    echo "Last lines before it switched off:"
    gcloud logging read "resource.type=gce_instance AND resource.labels.instance_id=$ID AND logName:serial_port_1_output" \
        --freshness=7d --limit=400 --format="value(textPayload)" | grep -a "startup-script:" | grep -av "^\[" | grep -aE "$WANTED" | sed "s/^.*startup-script: //" | head -n 12 | tac
    echo "If training has not finished, start it again with: bash $(dirname "$0")/create_vm.sh"
    echo "Checkpoints so far: https://huggingface.co/ahigazy1/adrenal-multiorgan-model"
fi
