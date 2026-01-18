#!/usr/bin/env bash

# Join split files
cat section_* > transfer.tar.xz

# Verify archive integrity
xz -t transfer.tar.xz

# Extract archive
tar -xJvf transfer.tar.xz
