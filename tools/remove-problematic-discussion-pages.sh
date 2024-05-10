#!/bin/bash
#
# Tool for removing problematic discussion pages (currently all pages except the first ones)
#
# Usage: ./remove-problematic-discussion-pages.sh [path-to-extended_output_data]

EXTERNAL_OUTPUT_DATA=${1:-"/tmp/knn/extended_output_data"}

for website_dir in "$EXTERNAL_OUTPUT_DATA"/*; do
    server_name=$(echo "$website_dir" | sed -E "s|$EXTERNAL_OUTPUT_DATA/||")
    echo "Processing server $server_name..."

    for article_dir in "$website_dir"/*; do
        for page_data in "$article_dir"/{bounding-boxes,hierarchy,screenshot,html}/*; do
            if [[ ! "$page_data" =~ /1\.[a-z]+$ ]]; then
                echo -e "\tRemoving $page_data..."
                rm -r "$page_data"
            fi
        done
    done
done
