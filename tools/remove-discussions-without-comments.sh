#!/bin/bash
#
# Tools for removing discussions without comments from raw data
#
# Usage: ./remove-discussions-without-comments.sh [path-to-extended_output_data]

EXTERNAL_OUTPUT_DATA=${1:-"/tmp/knn/extended_output_data"}

for website_dir in "$EXTERNAL_OUTPUT_DATA"/*; do
    server_name=$(echo "$website_dir" | sed -E "s|$EXTERNAL_OUTPUT_DATA/||")
    echo "Processing server $server_name..."

    for article_dir in "$website_dir"/*; do
        # Discussions without comments hasn't bounding boxes and hierarchy subfolders
        if [[ ! -d $article_dir/bounding-boxes ]] || [[ ! -d $article_dir/hierarchy ]]; then
            echo -e "\tRemoving $article_dir..."
            rm -r "$article_dir"
        fi
    done
done
