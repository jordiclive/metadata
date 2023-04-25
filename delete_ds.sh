dir="/fsx/home-jordiclive/tmp/metadata-quarter-html"

while true; do
    find "$dir" -type d -regex ".*/global_step[0-9]*" | while read -r folder; do
        step=$(basename "$folder" | sed -e 's/global_step//g')
        if ! (( step % 3000 == 0 )); then
            echo "Deleting: $folder"
            rm -rf "$folder"
        fi
    done

    # Sleep for 45 minutes
    sleep 2700
done