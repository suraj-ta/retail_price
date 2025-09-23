#!/bin/bash
set -e

# Clean up old clone if exists
rm -rf mlopsplatform

echo "📥 Cloning manifest repo..."
git clone https://$1@dev.azure.com/tamlopsplatform/mlopsplatform/_git/mlopsplatform
cd mlopsplatform
ls -lah
