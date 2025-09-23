#!/bin/bash
set -e
cd mlopsplatform

git config user.name "Devops pipeline"
git config user.email "nayana.kumari@tigeranalytics.com"

git add .
if git commit -m "Push from dev pipeline"; then
    echo "✅ Changes committed"
    git push
else
    echo "ℹ️ No changes to commit"
fi
